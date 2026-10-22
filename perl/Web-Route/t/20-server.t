use strict;
use warnings;
use Test::More;

use Web::Route;
use File::Temp qw(tempdir);
use IO::Socket::UNIX;
use Socket qw(SOCK_STREAM);
use JSON::PP ();
use POSIX ();
use Time::HiRes qw(time sleep);

# The server on a unix socket, in a child process; the client speaks raw HTTP/1.1 as nginx does
my $dir = tempdir(CLEANUP => 1);
my $sock = "$dir/s.sock";
my $flag = "$dir/gone";

route::get '/hello/:name' => sub { my ($r) = @_; "hello $r->{params}{name} $r->{args}{x}" };
route::post '/echo' => sub { my ($r) = @_; { json => $r->{json}, len => length $r->{body}, ct => $r->{headers}{'content-type'} } };
route::get '/events' => sub {
    stream(200, [ 'Content-Type' => 'text/event-stream' ], sub {
        my ($write) = @_;
        $write->(sse({ n => $_ })) and sleep 0.2 for 1 .. 3;
        $write->(sse('[DONE]'));
    });
};
route::get '/endless' => sub {
    stream(200, [ 'Content-Type' => 'text/event-stream' ], sub {
        my ($write) = @_;
        my $n = 0;
        while ($write->(sse({ n => ++$n }))) { sleep 0.01; last if $n > 2000 }
        open my $f, '>', $flag or die; print $f $n; close $f;  # what the handler saw
    });
};

my $pid = fork // die "fork: $!";
if (!$pid) {
    open STDERR, '>', "$dir/err" or die;
    run(listen => $sock, mode => 0600, timeout => 1, max_body => 1000);
    POSIX::_exit(0);
}
for (1 .. 100) { last if -S $sock; sleep 0.05 }
ok(-S $sock, 'the socket exists');
is((stat $sock)[2] & 0777, 0600, 'with the mode asked for');

sub conn { IO::Socket::UNIX->new(Peer => $sock, Type => SOCK_STREAM) or die "connect: $!" }
sub request {
    my ($raw) = @_;
    my $c = conn();
    print $c $raw;
    local $/;
    my $res = <$c>;
    my ($head, $body) = split /\r\n\r\n/, $res // '', 2;
    my ($status) = ($head // '') =~ m{^HTTP/1\.1 (\d{3})};
    my %h = map { /^([^:]+):\s*(.*)$/ ? (lc $1 => $2) : () } split /\r\n/, $head // '';
    return ($status, \%h, $body);
}
sub GET { request("GET $_[0] HTTP/1.1\r\nHost: x\r\n\r\n") }

my ($st, $h, $b) = GET('/hello/w%C3%B6rld?x=a+b%21');
is($st, 200, 'GET');
is($b, "hello w\xc3\xb6rld a b!", 'parameters and query, sent as UTF-8');
is($h->{'content-length'}, length $b, 'Content-Length');
is($h->{connection}, 'close', 'Connection: close');

my $json = '{"a":[1,2],"b":"é"}';
($st, $h, $b) = request("POST /echo HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: "
                        . length($json) . "\r\n\r\n$json");
is($st, 200, 'POST JSON');
is_deeply(JSON::PP->new->utf8->decode($b), { json => { a => [1, 2], b => "\x{e9}" }, len => length $json,
                                            ct => 'application/json' }, 'body and headers');
($st) = request("POST /echo HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: 5\r\n\r\n{nope");
is($st, 400, 'invalid JSON: 400');
($st) = request("POST /echo HTTP/1.1\r\nContent-Length: 2000\r\n\r\n" . ('x' x 2000));
is($st, 413, 'body too large: 413');
($st) = request("POST /echo HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n");
is($st, 501, 'chunked body: 501');
($st) = request("BLAH\r\n\r\n");
is($st, 400, 'bad request line: 400');
($st) = request("POST /echo HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc");
is($st, 408, 'a body that never comes: 408 after the timeout');
($st, $h) = request("HEAD /hello/x HTTP/1.1\r\n\r\n");
is($st, 200, 'HEAD');
($st, $h) = GET('/nope');
is($st, 404, 'no route: 404');

# Streaming: the first event arrives before the handler is done
my $c = conn();
print $c "GET /events HTTP/1.1\r\n\r\n";
my $t0 = time;
my $got = '';
while ($got !~ /"n":1/) { sysread($c, $got, 4096, length $got) or last }
my $first = time - $t0;
$got .= $_ while <$c>;
my $all = time - $t0;
ok($first < 0.15 && $all > 0.5, sprintf('first event after %.2f s, all after %.2f s', $first, $all));
like($got, qr/text\/event-stream.*X-Accel-Buffering: no/s, 'event stream headers');
like($got, qr/data: \{"n":1\}\n\ndata: \{"n":2\}\n\ndata: \{"n":3\}\n\ndata: \[DONE\]\n\n\z/, 'the events');

# A client that goes away: $write returns false and the handler stops
$c = conn();
print $c "GET /endless HTTP/1.1\r\n\r\n";
sysread($c, my $x, 100);
close $c;
for (1 .. 100) { last if -s $flag; sleep 0.05 }
open my $f, '<', $flag;
my $n = <$f> // 0;
ok($n > 0 && $n < 2000, "the handler saw the client go after $n events");

# SIGTERM: the server stops and removes its socket
kill TERM => $pid;
waitpid $pid, 0;
is($?, 0, 'stopped on SIGTERM');
ok(!-e $sock, 'the socket file is gone');

# host:port
my $port;
$pid = fork // die;
if (!$pid) {
    open STDERR, '>', "$dir/err2" or die;
    run(listen => '127.0.0.1:0', on_start => sub { open my $p, '>', "$dir/port"; print $p $_[0]->sockport; close $p });
    POSIX::_exit(0);
}
for (1 .. 100) { last if -s "$dir/port"; sleep 0.05 }
$port = do { open my $p, '<', "$dir/port"; <$p> };
require HTTP::Tiny;
my $res = HTTP::Tiny->new->get("http://127.0.0.1:$port/hello/tcp?x=1");
is($res->{content}, 'hello tcp 1', 'TCP with HTTP::Tiny');
kill TERM => $pid;
waitpid $pid, 0;

done_testing;
