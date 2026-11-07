use strict;
use warnings;
use Test::More;
use JSON::PP ();

use ROCm::Inference::Log;

# No GPU: the formatter alone, with a fake clock and an in-memory handle
my $now = 1_000.25;
my $clock = sub { $now };
sub log_to {
    my (%o) = @_;
    open my $fh, '>', \my $buf or die;
    my $log = ROCm::Inference::Log->new(%o, fh => $fh, clock => $clock);
    return ($log, sub { close $fh; $buf });
}

{
    my ($log, $out) = log_to(level => 'info', color => 'never');
    $log->info('request', user => 'alice', ip => '10.0.0.7', conv => 'conv-17', req => 'chatcmpl-6f3a',
               model => 'Qwen3.6', event => 'start', msg => 'stream, 14 msgs');
    my $line = $out->();
    like($line, qr/^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.250 +\d+ info /, 'timestamp with milliseconds, pid, level');
    like($line, qr/alice\s+10\.0\.0\.7\s+conv-17\s+chatcmpl-6f3a\s+Qwen3\.6\s+start\s+stream, 14 msgs\n\z/,
         'user, ip, conv, req, model, event, message in order');
    unlike($line, qr/\e\[/, 'no colour codes when color is never');
}
{
    my ($log, $out) = log_to(level => 'info', color => 'never');
    $log->info('model', model => 'M');
    like($out->(), qr/ info\s+(?:-\s+){4}M\s+-\s+/, 'missing fields show as -');
}
{
    my ($log, $out) = log_to(level => 'info', format => 'json', color => 'never');
    $log->info('prefill', user => 'bob', req => 'r1', model => 'M', event => 'prefill_end', msg => 'done 2532 tok');
    my $j = JSON::PP->new->decode($out->());
    is($j->{user}, 'bob', 'json: user');
    is($j->{kind}, 'prefill', 'json: kind');
    is($j->{event}, 'prefill_end', 'json: event');
    is($j->{msg}, 'done 2532 tok', 'json: msg');
    ok(!exists $j->{ip}, 'json: a missing field is left out');
    is($j->{pid}, $$, 'json: pid');
}
{
    my ($log, $out) = log_to(level => 'info', color => 'always');
    $log->info('request', msg => 'hi');
    like($out->(), qr/\e\[1mhi\e\[0m/, 'color always: request bold');
    my ($log2, $out2) = log_to(level => 'info', color => 'always');
    $log2->info('debug_kind_none', msg => 'plain');
    unlike($out2->(), qr/\e\[\d+mplain/, 'a kind without a colour is plain');
}
{
    my ($log, $out) = log_to(level => 'info', color => 'never');
    $log->debug('draft', msg => 'hidden');
    $log->info('model', msg => 'shown');
    my $s = $out->();
    like($s, qr/shown/, 'info passes at level info');
    unlike($s, qr/hidden/, 'debug is filtered at level info');
    my ($dbg, $dout) = log_to(level => 'debug', color => 'never');
    $dbg->debug('draft', msg => 'now here');
    like($dout->(), qr/now here/, 'debug passes at level debug');
    my ($none, $nout) = log_to(level => 'none', color => 'never');
    $none->error(msg => 'x');
    is($nout->() // '', '', 'level none writes nothing, not even errors');
    my ($warn, $wout) = log_to(level => 'warn', color => 'never');
    $warn->info('model', msg => 'no');
    $warn->warn_(msg => 'yes');
    my $w = $wout->();
    like($w, qr/yes/, 'warn passes at level warn');
    unlike($w, qr/\bno\b/, 'info is filtered at level warn');
}
{
    my ($log, $out) = log_to(level => 'info', color => 'never');
    my $total = 1000;
    my @at;
    for my $done (1 .. 1000) {
        $now = 1_000 + $done * 0.001;  # one millisecond per token: the 2 s rule never fires here
        push @at, $done if $log->progress('r', $done, $total, msg => '');
    }
    is_deeply(\@at, [1, 101, 201, 301, 401, 501, 601, 701, 801, 901, 1000],
              'progress: first, every 10%, and the last');
    $now = 2_000;
    my ($slow, $sout) = log_to(level => 'info', color => 'never');
    my @slow;
    for my $done (1 .. 5) {
        $now = 2_000 + $done;  # a second per step: a line at least every 2 s
        push @slow, $done if $slow->progress('s', $done, 1000, msg => '');
    }
    is_deeply(\@slow, [1, 3, 5], 'progress: a line every 2 s when the total moves slowly');
    like($sout->(), qr/prefill 0%/, 'progress reports the percentage');
}
{
    my ($log, $out) = log_to(level => 'info', color => 'never');
    $now = 3_000;
    $log->progress('p', 5, 10);
    $log->progress('p', 10, 10);
    ok(!exists $log->{progress}{p}, 'the progress state is dropped at the last line');
}
{
    ok(!eval { ROCm::Inference::Log->new(level => 'loud'); 1 }, 'a bad level croaks');
    ok(!eval { ROCm::Inference::Log->new(format => 'xml'); 1 }, 'a bad format croaks');
    ok(!eval { ROCm::Inference::Log->new(color => 'maybe'); 1 }, 'a bad color croaks');
    my $auto = ROCm::Inference::Log->new(fh => do { open my $fh, '>', \my $b; $fh });
    ok(!$auto->{color}, 'auto colour is off for a handle that is not a terminal');
}
done_testing;
