package ROCm::Inference::Log;

use strict;
use warnings;
use Carp qw(croak);
use JSON::PP ();
use POSIX ();
use Time::HiRes ();

# The server's log lines (WORKLOG logging plan, Phase 3): one line per event, with the PID, the user, the client
# address, the conversation, the request, the model and the event; text (fixed columns, so docker logs stays
# greppable) or JSON lines. Colours by kind, off unless the output is a terminal (or color => 'always').

our %LEVEL = (none => -1, error => 0, warn => 1, info => 2, debug => 3);
our %COLOR = (request => '1', prefill => '36', generation => '32', model => '35', warn => '33', error => '31');

# new(level => 'info', format => 'text'|'json', color => 'auto'|'always'|'never', fh => \*STDERR, clock => sub { time })
# auto colours a terminal unless NO_COLOR is set; always colours regardless; never does not.
sub new {
    my ($class, %o) = @_;
    my $level = $o{level} // 'info';
    my $format = $o{format} // 'text';
    my $color = $o{color} // 'auto';
    croak "ROCm::Inference::Log: level must be one of " . join(', ', sort keys %LEVEL) unless exists $LEVEL{$level};
    croak "ROCm::Inference::Log: format must be text or json" unless $format =~ /\A(?:text|json)\z/;
    croak "ROCm::Inference::Log: color must be auto, always or never" unless $color =~ /\A(?:auto|always|never)\z/;
    my $fh = $o{fh} // \*STDERR;
    my $colored = $color eq 'always' ? 1
                : $color eq 'never' ? 0
                : (-t $fh && !defined $ENV{NO_COLOR}) ? 1 : 0;
    return bless {
        level => $level, format => $format, color => $colored, fh => $fh,
        clock => $o{clock} // sub { Time::HiRes::time() }, progress => {},
    }, $class;
}

sub level { $_[0]{level} }

# Whether a line at $level is written (the level filter: 'info' writes error, warn and info)
sub enabled {
    my ($self, $level) = @_;
    croak "ROCm::Inference::Log: no level $level" unless exists $LEVEL{$level};
    return $LEVEL{$self->{level}} >= $LEVEL{$level};
}

# emit($level, $kind, %fields): fields user, ip, conv, req, model, event, msg. kind colours the line (see %COLOR).
sub emit {
    my ($self, $level, $kind, %f) = @_;
    return 0 unless $self->enabled($level);
    my $t = $self->{clock}->();
    my $ts = POSIX::strftime('%Y-%m-%d %H:%M:%S', localtime $t) . sprintf('.%03d', 1000 * ($t - int $t));
    my $line;
    if ($self->{format} eq 'json') {
        my %j = (ts => $ts, pid => $$, level => $level, kind => $kind);
        $j{$_} = $f{$_} for grep { defined $f{$_} } qw(user ip conv req model event);
        $j{msg} = $f{msg} // '';
        $line = JSON::PP->new->canonical->encode(\%j);
    } else {
        my $msg = $f{msg} // '';
        $msg = "\e[$COLOR{$kind}m$msg\e[0m" if $self->{color} && $COLOR{$kind};
        $line = sprintf '%s %6d %-5s %-10s %-39s %-6s %-22s %-24s %-16s %s',
            $ts, $$, $level, _col($f{user}), _col($f{ip}), _col($f{conv}), _col($f{req}),
            _col($f{model}), _col($f{event}), $msg;
    }
    print { $self->{fh} } "$line\n";
    return 1;
}

# a missing field shows as -; a value is never cut (a request id or a model name is what the reader looks for)
sub _col { my ($v) = @_; $v = '-' unless defined $v && length $v; $v =~ s/\s+/_/g; $v }

sub info  { my ($self, $kind, %f) = @_; $self->emit('info', $kind, %f) }
sub debug { my ($self, $kind, %f) = @_; $self->emit('debug', $kind, %f) }
sub warn_ { my ($self, %f) = @_; $self->emit('warn', 'warn', %f) }
sub error { my ($self, %f) = @_; $self->emit('error', 'error', %f) }

# progress($key, $done, $total, %fields): a progress line for a long job (a prefill), at most every 10% of the total
# or 2 s, whichever comes first; the first and the last always. $key names the job (its request id).
sub progress {
    my ($self, $key, $done, $total, %f) = @_;
    my $now = $self->{clock}->();
    my $st = $self->{progress}{$key} //= { done => undef, t => 0 };
    my $last = $done >= $total;
    my $due = !defined $st->{done} || $last || $done - $st->{done} >= $total / 10 || $now - $st->{t} >= 2;
    return 0 unless $due;
    $st->{done} = $done;
    $st->{t} = $now;
    delete $self->{progress}{$key} if $last;
    my $pct = $total > 0 ? int(100 * $done / $total) : 100;
    return $self->emit('info', 'prefill', %f, msg => sprintf('prefill %d%%  %d/%d tok', $pct, $done, $total));
}

1;

__END__

=head1 NAME

ROCm::Inference::Log - the server's log lines: text or JSON, coloured by kind, throttled progress

=head1 SYNOPSIS

    my $log = ROCm::Inference::Log->new(level => 'info', format => 'text', color => 'auto');
    $log->info('request', user => 'alice', ip => '10.0.0.7', req => 'chatcmpl-1', model => 'Qwen3.6',
               msg => 'POST /v1/chat/completions, stream');
    $log->progress('chatcmpl-1', 1040, 2532, model => 'Qwen3.6', msg => '');

=head1 DESCRIPTION

Every line has a timestamp (milliseconds), the PID, the level, and the fields C<user>, C<ip>, C<conv>, C<req>,
C<model>, C<event> (missing ones show as C<->), then the message. C<format =E<gt> 'json'> writes the same as one JSON
object per line. C<level> filters: C<none>, C<error>, C<warn>, C<info> (the default), C<debug>.

C<color>: C<auto> colours a terminal unless C<NO_COLOR> is set; C<always> and C<never> do as they say. Colours by kind:
request bold, prefill cyan, generation green, model magenta, warn yellow, error red.

=head1 METHODS

=over

=item new(%options)

C<level>, C<format> (text | json), C<color> (auto | always | never), C<fh> (default STDERR), C<clock> (a code ref
returning the time in seconds; for tests). Croaks on a bad value.

=item emit($level, $kind, %fields)

One line. C<info>, C<debug>, C<error> and C<warn_> are shorthands for it.

=item progress($key, $done, $total, %fields)

A C<prefill> line with the percentage, written when at least 10% of the total is new since the last line for C<$key>
or 2 seconds have passed, and always for the first and the last. Returns true when a line was written.

=item enabled($level)

Whether a line at that level is written.

=back

=head1 SEE ALSO

L<ROCm::Inference>, the server's C<--log-level>, C<--log-format>, C<--log-color> options.

=cut
