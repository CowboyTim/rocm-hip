use strict;
use warnings;
use Test::More;

use Web::Route;
use File::Temp qw(tempdir);
use IO::Socket::UNIX;
use Socket qw(SOCK_STREAM);
use POSIX ();
use Time::HiRes qw(time sleep);

# pull responses: served together by run's loop, fed by a tick (as an inference server steps its sequences), and
# on_close when a client leaves early
my $dir = tempdir(CLEANUP => 1);
my $sock = "$dir/s.sock";

my %jobs;  # id => { n, out }
my $ticks = 0;
route::get '/count/:id/:n' => sub {
    my ($r) = @_;
    my $id = $r->{params}{id};
    $jobs{$id} = { n => $r->{params}{n}, out => [] };
    pull(200, [ 'Content-Type' => 'text/event-stream' ], sub {
        my $j = $jobs{$id} or return undef;
        return join '', map { sse({ id => $id, i => $_ }) } splice @{ $j->{out} } if @{ $j->{out} };
        return undef if $j->{n} <= 0;
        return '';
    }, on_close => sub { open my $f, '>', "$dir/closed-$id" or die; close $f; delete $jobs{$id} });
};
route::get '/ticks' => sub { { ticks => $ticks } };

my $pid = fork // die "fork: $!";
if (!$pid) {
    open STDERR, '>', "$dir/err" or die;
    # each tick: one piece for every job (the tick does the work the responses pull)
    run(listen => $sock, timeout => 1, tick => sub {
        ++$ticks;
        for my $j (values %jobs) {
            next if $j->{n} <= 0;
            push @{ $j->{out} }, $j->{n}--;
        }
        sleep 0.002;
        return 1;
    });
    POSIX::_exit(0);
}
for (1 .. 200) { last if -S $sock; sleep 0.01 }

sub connect_get {
    my ($path) = @_;
    my $c = IO::Socket::UNIX->new(Peer => $sock, Type => SOCK_STREAM) or die "connect: $!";
    print $c "GET $path HTTP/1.1\r\nHost: x\r\n\r\n";
    return $c;
}

# two slow responses at once: both complete, each piece in order, and they overlap in time
my $t0 = time;
my ($a, $b) = (connect_get('/count/a/50'), connect_get('/count/b/50'));
my %body;
for my $p ([a => $a], [b => $b]) { local $/; $body{ $p->[0] } = readline $p->[1] }
my $dt = time - $t0;
for my $id (qw(a b)) {
    my @i = $body{$id} =~ /"i":(\d+)/g;
    is("@i", join(' ', reverse 1 .. 50), "$id: all 50 pieces in order");
    like($body{$id}, qr{^HTTP/1\.1 200 OK.*Content-Type: text/event-stream}s, "$id: an event stream");
}
ok($dt < 0.002 * 50 * 2 + 1, sprintf('served together (%.2f s)', $dt));

# a plain request is answered while a pull is open
my $slow = connect_get('/count/c/100000');
sleep 0.05;
my $q = connect_get('/ticks');
my $resp = do { local $/; readline $q };
like($resp, qr/"ticks":\d+/, 'a plain request is served while a pull is open');

# the slow client leaves: on_close runs
close $slow;
for (1 .. 200) { last if -e "$dir/closed-c"; sleep 0.01 }
ok(-e "$dir/closed-c", 'on_close when the client leaves early');
ok(!-e "$dir/closed-a", 'not for a response that completed');

kill TERM => $pid;
waitpid $pid, 0;
ok(!-e $sock, 'SIGTERM: stops, socket removed');
is(do { local (@ARGV, $/) = "$dir/err"; <> } // '', '', 'no warnings');
done_testing;
