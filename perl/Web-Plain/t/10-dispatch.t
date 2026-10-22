use strict;
use warnings;
use utf8;
use Test::More;

use Web::Plain;
use Web::Plain qw(dispatch routes clear_routes);
use JSON::PP ();

# Routing without a socket: dispatch on request hashes
sub req { my ($method, $path, %more) = @_; return { method => $method, path => $path, args => {}, %more } }
sub body { my ($res) = @_; ref $res->[2] ? join('', @{ $res->[2] }) : $res->[2] }
sub hdr { my ($res, $name) = @_; my %h = @{ $res->[1] }; $h{$name} }
my $J = JSON::PP->new->utf8;

get '/' => sub { 'root' };
get '/health' => sub { { ok => 1 } };
get '/users/:id' => sub { my ($r) = @_; "user $r->{params}{id}" };
get '/users/:id/posts/:post' => sub { my ($r) = @_; [ 200, [], "$r->{params}{id}/$r->{params}{post}" ] };
get '/files/*rest' => sub { my ($r) = @_; text($r->{params}{rest}) };
post '/users' => sub { my ($r) = @_; json({ created => $r->{json}{name} }, 201) };
del '/users/:id' => sub { undef };
any '/any' => sub { my ($r) = @_; $r->{method} };
get '/list' => sub { [1, 2, 3] };
get '/die' => sub { die "boom\n" };
get '/teapot' => sub { die text('no coffee', 418) };
get '/utf8/:w' => sub { my ($r) = @_; "→ $r->{params}{w}" };

is(body(dispatch(req(GET => '/'))), 'root', 'a string is text');
is(hdr(dispatch(req(GET => '/')), 'Content-Type'), 'text/plain; charset=utf-8', 'text content type');
is_deeply($J->decode(body(dispatch(req(GET => '/health')))), { ok => 1 }, 'a hash is JSON');
is(hdr(dispatch(req(GET => '/health')), 'Content-Type'), 'application/json', 'JSON content type');
is(body(dispatch(req(GET => '/users/42'))), 'user 42', ':name parameter');
is(body(dispatch(req(GET => '/users/7/posts/x%20y'))), '7/x y', 'two parameters, URL-decoded');
is(body(dispatch(req(GET => '/files/a/b/c.txt'))), 'a/b/c.txt', '*name takes the rest');
my $r = dispatch(req(POST => '/users', json => { name => 'tim' }));
is($r->[0], 201, 'status from json()');
is_deeply($J->decode(body($r)), { created => 'tim' }, 'request JSON');
is(dispatch(req(DELETE => '/users/1'))->[0], 204, 'undef is 204');
is(body(dispatch(req(PATCH => '/any'))), 'PATCH', 'any method');
is_deeply($J->decode(body(dispatch(req(GET => '/list')))), [1, 2, 3], 'an array (not a response) is JSON');
is(dispatch(req(HEAD => '/health'))->[0], 200, 'HEAD on a GET route');

$r = dispatch(req(POST => '/health'));
is($r->[0], 405, 'wrong method: 405');
is(hdr($r, 'Allow'), 'GET, HEAD', 'with Allow');
is(dispatch(req(GET => '/nope'))->[0], 404, 'no route: 404');
is(dispatch(req(GET => '/users/1/x'))->[0], 404, ':name does not span segments');

{
    my $warned = '';
    local $SIG{__WARN__} = sub { $warned .= $_[0] };
    $r = dispatch(req(GET => '/die'));
    is($r->[0], 500, 'a die is 500');
    unlike(body($r), qr/boom/, 'the message is not sent');
    like($warned, qr/GET \/die: boom/, 'it goes to STDERR');
}
$r = dispatch(req(GET => '/teapot'));
is_deeply([ $r->[0], body($r) ], [ 418, 'no coffee' ], 'die with a response');
is(body(dispatch(req(GET => '/utf8/%C3%A9t%C3%A9'))), '→ été', 'UTF-8 parameters decode to characters');

is_deeply([ map { "@$_" } routes() ], [ 'GET /', 'GET /health', 'GET /users/:id', 'GET /users/:id/posts/:post',
          'GET /files/*rest', 'POST /users', 'DELETE /users/:id', '* /any', 'GET /list', 'GET /die',
          'GET /teapot', 'GET /utf8/:w' ], 'routes()');
clear_routes();
is(dispatch(req(GET => '/'))->[0], 404, 'clear_routes');

is(sse('hi'), "data: hi\n\n", 'sse: text');
is(sse({ a => 1 }, event => 'x', id => 3), qq[event: x\nid: 3\ndata: {"a":1}\n\n], 'sse: JSON with fields');
is(sse("a\nb"), "data: a\ndata: b\n\n", 'sse: multi-line');
is(sse('é'), "data: \xc3\xa9\n\n", 'sse: UTF-8 bytes');

ok(!eval { get 'nope' => sub {}; 1 }, 'a pattern must start with /');

done_testing;
