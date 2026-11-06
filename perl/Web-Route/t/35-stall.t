use strict;
use warnings;
use Test::More;

use Web::Route;
use File::Temp qw(tempdir);
use IO::Socket::UNIX;
use Socket qw(SOCK_STREAM);
use POSIX ();
use Time::HiRes qw(time sleep);

# a pull response whose client takes no bytes is closed off (on_close runs, as when the client leaves); a client
# that reads, or a response with nothing to send yet, is not. Two limits, each tested on its own server: output
# past max_pending bytes, and no bytes taken for stall seconds.
my $dir = tempdir(CLEANUP => 1);

my %job;  # id => { kind, n, out }
route::get '/job/:id/:kind/:n' => sub {
    my ($r) = @_;
    my ($id, $kind) = @{ $r->{params} }{qw(id kind)};
    $job{$id} = { kind => $kind, n => $r->{params}{n}, out => [] };
    pull(200, [ 'Content-Type' => 'text/event-stream' ], sub {
        my $j = $job{$id} or return undef;
        return join '', map { sse($_) } splice @{ $j->{out} } if @{ $j->{out} };
        return undef if $j->{kind} eq 'finite' && $j->{n} <= 0;
        return '';
    }, on_close => sub { open my $f, '>', "$dir/closed-$id" or die; close $f; delete $job{$id} });
};

my $c_endless = 0;
sub start_server {
    my ($sock, %limits) = @_;
    my $pid = fork // die "fork: $!";
    return ($pid) if $pid;
    open STDERR, '>', "$sock.err" or die;
    # one piece per round for an endless job, a few for a finite one; the quiet job never sends anything
    run(listen => $sock, timeout => 1, tick => sub {
        for my $j (values %job) {
            if ($j->{kind} eq 'endless') { push @{ $j->{out} }, { i => ++$c_endless, pad => 'x' x 65536 } }
            elsif ($j->{kind} eq 'finite' && $j->{n} > 0) { push @{ $j->{out} }, { i => $j->{n}-- } }
        }
        sleep 0.002;
        return 1;
    }, %limits);
    POSIX::_exit(0);
}

# the byte limit: 64 KB unsent closes the response; the time limit (1 s) never fires first
my $sock_b = "$dir/bytes.sock";
my $pid_b = start_server($sock_b, max_pending => 64 * 1024, stall => 60);
# the time limit: no byte limit to speak of, 1 s without the client taking a byte closes the response
my $sock_t = "$dir/time.sock";
my $pid_t = start_server($sock_t, max_pending => 1 << 40, stall => 1);
for my $s ($sock_b, $sock_t) { for (1 .. 200) { last if -S $s; sleep 0.01 } }

sub connect_get {
    my ($sock, $path) = @_;
    my $c = IO::Socket::UNIX->new(Peer => $sock, Type => SOCK_STREAM) or die "connect: $!";
    print $c "GET $path HTTP/1.1\r\nHost: x\r\n\r\n";
    return $c;
}
sub wait_for { my ($f) = @_; for (1 .. 300) { return 1 if -e $f; sleep 0.01 } return 0 }

# a client that reads nothing and keeps its socket open: closed off, on_close runs
for my $case ([ $sock_b, 'bytes', 'the byte limit' ], [ $sock_t, 'time', 'the time limit' ]) {
    my ($sock, $id, $what) = @$case;
    my $c = connect_get($sock, "/job/$id/endless/0");
    sysread($c, my $head, 200);
    like($head, qr{^HTTP/1\.1 200 OK}, "$what: the stalled client got its response");
    ok(wait_for("$dir/closed-$id"), "$what: a client that takes no bytes is closed off (on_close)");
    close $c;
}

# a response with nothing to send yet stays open while it waits
my $quiet = connect_get($sock_t, '/job/q/quiet/0');
sleep 3;
ok(!-e "$dir/closed-q", 'a quiet response is not closed off for having nothing to send');
close $quiet;
ok(wait_for("$dir/closed-q"), 'and on_close runs when that client leaves');

# a client that reads everything: complete, not closed off
my $reader = connect_get($sock_t, '/job/f/finite/3000');
my $body = do { local $/; readline $reader };
my @i = $body =~ /"i":(\d+)/g;
is(scalar @i, 3000, 'a reading client gets all 3000 pieces');
ok(!-e "$dir/closed-f", 'a reading client is not closed off');

for my $pid ($pid_b, $pid_t) { kill TERM => $pid; waitpid $pid, 0 }
is(do { local (@ARGV, $/) = "$dir/bytes.sock.err"; <> } // '', '', 'no warnings (bytes)');
is(do { local (@ARGV, $/) = "$dir/time.sock.err"; <> } // '', '', 'no warnings (time)');
done_testing;
