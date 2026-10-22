package Web::Route;

use strict;
use warnings;

use Carp qw(croak);
use Exporter 'import';
use IO::Select;
use IO::Socket::UNIX;
use Socket qw(SOCK_STREAM);
use Time::HiRes ();
use JSON::PP ();

our $VERSION = '0.01';

our @EXPORT    = qw(json text stream sse run);
our @EXPORT_OK = qw(dispatch);

# The route table: [method or undef for any, pattern, regex, [param names], handler]
my @ROUTES;

my $JSON = JSON::PP->new->utf8->canonical->allow_nonref->allow_blessed->convert_blessed;

my %REASON = (
    200 => 'OK', 201 => 'Created', 202 => 'Accepted', 204 => 'No Content', 301 => 'Moved Permanently',
    302 => 'Found', 304 => 'Not Modified', 400 => 'Bad Request', 401 => 'Unauthorized', 403 => 'Forbidden',
    404 => 'Not Found', 405 => 'Method Not Allowed', 408 => 'Request Timeout', 409 => 'Conflict',
    411 => 'Length Required', 413 => 'Content Too Large', 415 => 'Unsupported Media Type',
    422 => 'Unprocessable Content', 429 => 'Too Many Requests', 431 => 'Request Header Fields Too Large',
    500 => 'Internal Server Error', 501 => 'Not Implemented', 502 => 'Bad Gateway',
    503 => 'Service Unavailable', 504 => 'Gateway Timeout',
);

# Adds a route: $method undef for any method; :name in the pattern matches one path segment, *name the rest
# of the path, both land in $req->{params}
sub _add {
    my ($method, $pattern, $handler) = @_;
    croak "Web::Route: a route needs a pattern starting with / and a code reference"
        unless defined $pattern && $pattern =~ m{^/} && ref $handler eq 'CODE';
    my @names;
    my $re = join '', map {
        /^:(\w+)\z/ ? do { push @names, $1; '([^/]+)' }
      : /^\*(\w+)\z/ ? do { push @names, $1; '(.*)' }
      : quotemeta
    } split m{(:\w+|\*\w+)}, $pattern;
    push @ROUTES, [ defined $method ? uc $method : undef, $pattern, qr/\A$re\z/, \@names, $handler ];
    return;
}

# The route declarations live in the route:: package, called by their full name, so nothing is exported
# that could clash (get, any, delete): route::get '/path' => sub { ... }. Assigned to globs, as subs named
# delete in a package would be ambiguous with the builtin inside it.
{
    no strict 'refs';
    for my $m (qw(get post put patch delete head options)) {
        my $method = uc $m;
        *{"route::$m"} = sub { _add($method, @_) };
    }
    *{'route::any'}   = sub { _add(undef, @_) };
    *{'route::add'}   = \&_add;
    *{'route::list'}  = sub { map { [ $_->[0] // '*', $_->[1] ] } @ROUTES };
    *{'route::clear'} = sub { @ROUTES = () };
}

# Responses: [status, [headers], body], body a string, an array of strings or a sub that streams
sub json {
    my ($data, $status, @headers) = @_;
    return [ $status // 200, [ 'Content-Type' => 'application/json', @headers ], $JSON->encode($data) ];
}

sub text {
    my ($text, $status, @headers) = @_;
    return [ $status // 200, [ 'Content-Type' => 'text/plain; charset=utf-8', @headers ], $text ];
}

# stream($status, [headers], sub { my ($write) = @_; ... }): $write->($text) sends a piece right away and
# returns false once the client is gone
sub stream {
    my ($status, $headers, $code) = @_;
    croak "Web::Route::stream: needs a code reference" unless ref $code eq 'CODE';
    return [ $status // 200, $headers // [ 'Content-Type' => 'application/octet-stream' ], $code ];
}

# sse($data, %fields): one server-sent event; $data a string or a structure (sent as JSON), fields event / id /
# retry
sub sse {
    my ($data, %f) = @_;
    $data = $JSON->encode($data) if ref $data;
    utf8::encode($data) if utf8::is_utf8($data);
    my $ev = '';
    $ev .= "$_: $f{$_}\n" for grep { defined $f{$_} } qw(event id retry);
    $ev .= "data: $_\n" for split /\n/, $data, -1;
    return "$ev\n";
}

# The handler's return value as [status, [headers], body]: a response as is, a hash (or an array of non-
# responses) as JSON, anything else as text
sub _response {
    my ($r) = @_;
    return [ 204, [], '' ] unless defined $r;
    return $r if ref $r eq 'ARRAY' && @$r == 3 && $r->[0] =~ /^\d{3}\z/ && ref $r->[1] eq 'ARRAY';
    return json($r) if ref $r eq 'HASH' || ref $r eq 'ARRAY';
    return text("$r");
}

# dispatch(\%req): the response to a parsed request (also for tests without a socket)
sub dispatch {
    my ($req) = @_;
    my ($method, $path) = ($req->{method}, $req->{path});
    my $head = $method eq 'HEAD';
    my %allow;
    for my $r (@ROUTES) {
        my @m = $path =~ $r->[2] or next;
        my $want = $r->[0];
        if (defined $want && $want ne $method && !($head && $want eq 'GET')) {
            $allow{$want} = 1;
            next;
        }
        my %p;
        @p{ @{ $r->[3] } } = map { _unescape($_) } @m;
        $req->{params} = \%p;
        my $res = eval { _response($r->[4]->($req)) };
        return $res if $res;
        my $err = $@ || 'unknown error';
        return $err if ref $err eq 'ARRAY';  # a handler may die with a response
        warn "Web::Route: $method $path: $err";
        return json({ error => 'internal server error' }, 500);
    }
    if (%allow) {
        $allow{HEAD} = 1 if $allow{GET};
        return json({ error => 'method not allowed' }, 405, Allow => join ', ', sort keys %allow);
    }
    return json({ error => 'not found' }, 404);
}

sub _unescape {
    my ($s) = @_;
    $s =~ s/%([0-9A-Fa-f]{2})/chr hex $1/ge;
    utf8::decode($s);
    return $s;
}

# The query string as a hash: a repeated key gives an array
sub _args {
    my ($q) = @_;
    my %a;
    for (split /[&;]/, $q // '') {
        next unless length;
        my ($k, $v) = map { tr/+/ /r } split /=/, $_, 2;
        ($k, $v) = (_unescape($k), _unescape($v // ''));
        if (exists $a{$k}) { $a{$k} = [ ref $a{$k} ? @{ $a{$k} } : $a{$k}, $v ] } else { $a{$k} = $v }
    }
    return \%a;
}

# Reads one request from $c: (\%req) or (undef, $status, $message)
sub _read_request {
    my ($c, $o) = @_;
    my $sel = IO::Select->new($c);
    my $buf = '';
    # more bytes: (1), or (undef, $status, $message) on a timeout or an early end
    my $read = sub {
        return (undef, 408, 'request timeout') unless $sel->can_read($o->{timeout});
        return sysread($c, $buf, 65536, length $buf) ? 1 : (undef, length $buf ? 400 : 0, 'incomplete request');
    };
    my $end;
    until (($end = index($buf, "\r\n\r\n")) >= 0) {
        return (undef, 431, 'request header too large') if length $buf > $o->{max_header};
        my @r = $read->();
        return @r unless $r[0];
    }
    my ($line, @lines) = split /\r\n/, substr($buf, 0, $end);
    substr($buf, 0, $end + 4, '');
    my ($method, $target, $proto) = $line =~ m{^([A-Z]+) (\S+) (HTTP/1\.[01])\z}
        or return (undef, 400, 'bad request line');
    my %h;
    for (@lines) {
        my ($k, $v) = /^([^:\s]+):\s*(.*?)\s*\z/ or return (undef, 400, 'bad header');
        $k = lc $k;
        $h{$k} = exists $h{$k} ? "$h{$k}, $v" : $v;
    }
    return (undef, 501, 'chunked request bodies are not supported (nginx: proxy_request_buffering on)')
        if ($h{'transfer-encoding'} // '') =~ /chunked/i;
    my $len = $h{'content-length'} // 0;
    return (undef, 400, 'bad content-length') unless $len =~ /^\d+\z/;
    return (undef, 413, 'request body too large') if $len > $o->{max_body};
    while (length $buf < $len) {
        my @r = $read->();
        return @r unless $r[0];
    }
    my ($path, $query) = split /\?/, $target, 2;
    my %req = (method => $method, path => $path, query => $query // '', protocol => $proto, headers => \%h,
               body => substr($buf, 0, $len), remote => $h{'x-real-ip'} // $h{'x-forwarded-for'});
    $req{args} = _args($req{query});
    if (length $req{body} && ($h{'content-type'} // '') =~ m{^application/(?:[\w.+-]+\+)?json}i) {
        $req{json} = eval { $JSON->decode($req{body}) };
        return (undef, 400, 'invalid JSON body') if $@;
    }
    return (\%req);
}

# Writes all of $data, false when the client is gone
sub _send {
    my ($c, $data) = @_;
    utf8::encode($data) if utf8::is_utf8($data);
    my $off = 0;
    while ($off < length $data) {
        my $n = syswrite($c, $data, length($data) - $off, $off);
        return 0 unless $n;
        $off += $n;
    }
    return 1;
}

sub _respond {
    my ($c, $req, $res) = @_;
    my ($status, $headers, $body) = @$res;
    my @h = @$headers;
    my %has = map { lc $h[ 2 * $_ ] => 1 } 0 .. $#h / 2;
    my $head = "HTTP/1.1 $status " . ($REASON{$status} // 'Unknown') . "\r\n";
    my $streaming = ref $body eq 'CODE';
    if ($streaming) {
        # no length: the end of the connection ends the body; nginx must not buffer an event stream
        push @h, 'X-Accel-Buffering' => 'no', 'Cache-Control' => 'no-cache'
            if !$has{'x-accel-buffering'} && grep { /^text\/event-stream/i } @h[ map { 2 * $_ + 1 } 0 .. $#h / 2 ];
    } else {
        $body = join '', @$body if ref $body eq 'ARRAY';
        $body //= '';
        utf8::encode($body) if utf8::is_utf8($body);
        push @h, 'Content-Length' => length $body unless $has{'content-length'} || $status == 204 || $status == 304;
    }
    push @h, Connection => 'close';
    for (my $i = 0; $i < @h; $i += 2) {
        (my $v = $h[ $i + 1 ]) =~ s/[\r\n]//g;
        $head .= "$h[$i]: $v\r\n";
    }
    _send($c, "$head\r\n") or return;
    return if $req && $req->{method} eq 'HEAD';
    if ($streaming) {
        my $ok = 1;
        eval { $body->(sub { $ok &&= _send($c, $_[0] // '') }); 1 }
            or warn "Web::Route: $req->{method} $req->{path}: stream: $@";
    } else {
        _send($c, $body);
    }
}

# The socket to listen on: a path (unix socket, default) or host:port
sub _listen {
    my ($o) = @_;
    my $l = $o->{listen} // croak "Web::Route::run: listen => '/path/to.sock' or 'host:port'";
    if ($l =~ m{^(?:\[([^\]]+)\]|([^/:]+)):(\d+)\z}) {
        require IO::Socket::IP;
        return IO::Socket::IP->new(LocalHost => $1 // $2, LocalPort => $3, Listen => $o->{backlog},
                                   ReuseAddr => 1, Type => SOCK_STREAM)
            // croak "Web::Route::run: cannot listen on $l: $@";
    }
    unlink $l if -S $l;
    my $s = IO::Socket::UNIX->new(Local => $l, Listen => $o->{backlog}, Type => SOCK_STREAM)
        // croak "Web::Route::run: cannot listen on $l: $!";
    chmod $o->{mode}, $l or croak "Web::Route::run: chmod $l: $!" if defined $o->{mode};
    return $s;
}

# run(listen => $socket, %options): serves the routes, one request at a time, until SIGINT / SIGTERM
sub run {
    my %o = (backlog => 128, timeout => 30, max_header => 65536, max_body => 16 << 20, log => 0, @_);
    my $srv = _listen(\%o);
    my $stop = 0;
    local $SIG{PIPE} = 'IGNORE';
    local $SIG{INT} = local $SIG{TERM} = sub { $stop = 1 };
    $o{on_start}->($srv) if $o{on_start};
    while (!$stop) {
        my $c = $srv->accept or next;  # (a signal ends the wait)
        my $t0 = Time::HiRes::time();
        my ($req, $status, $msg) = _read_request($c, \%o);
        my $res = $req ? dispatch($req) : $status ? json({ error => $msg }, $status) : undef;
        _respond($c, $req, $res) if $res;
        close $c;
        printf STDERR "%s %s %s %d %.0f ms\n", scalar localtime, $req ? ($req->{method}, $req->{path}) : ('-', '-'),
            $res->[0], 1000 * (Time::HiRes::time() - $t0) if $o{log} && $res;
    }
    close $srv;
    unlink $o{listen} if $o{listen} =~ m{/} && -S $o{listen};
    return;
}

1;

__END__

=head1 NAME

Web::Route - Flask-like routes and a small HTTP server for behind nginx, in plain Perl

=head1 SYNOPSIS

    use Web::Route;

    route::get '/health' => sub { { ok => 1 } };              # a hash: JSON
    route::get '/hello/:name' => sub {
        my ($req) = @_;
        "Hello, $req->{params}{name}!";                       # a string: text
    };
    route::post '/echo' => sub {
        my ($req) = @_;
        json({ got => $req->{json}, args => $req->{args} }, 201);
    };
    route::delete '/items/:id' => sub { undef };              # undef: 204
    route::get '/events' => sub {
        stream(200, ['Content-Type' => 'text/event-stream'], sub {
            my ($write) = @_;
            for my $n (1 .. 3) {
                $write->(sse({ n => $n })) or return;         # false: the client is gone
            }
            $write->(sse('[DONE]'));
        });
    };

    run(listen => '/run/app/app.sock', mode => 0660);

and in nginx:

    upstream app { server unix:/run/app/app.sock; }
    server {
        location / {
            proxy_pass http://app;
            proxy_http_version 1.1;
            proxy_buffering off;                              # stream responses as they come
            proxy_read_timeout 600s;
        }
    }

=head1 DESCRIPTION

Routes are declared with plain functions in the C<route::> package, requests are plain hashes and responses
plain arrays: no objects, no dependencies beyond core Perl. The route functions are called by their full name
(C<route::get>), so C<use Web::Route> exports nothing that clashes with other modules' C<get> or C<any>; it
exports the response helpers C<json>, C<text>, C<stream>, C<sse> and C<run>. The server behind them is deliberately small: it listens on a unix socket (or
C<host:port>), serves one request at a time and closes the connection after each, and leaves the hard work to
nginx in front of it: TLS, keep-alive, slow clients, limits, access control, static files. That suits a
process that holds one expensive resource (a model on a GPU, say) and does one thing at a time; requests that
arrive meanwhile wait in the socket's backlog.

=head2 Routes

=over

=item route::get / route::post / route::put / route::patch / route::delete / route::head / route::options / route::any ($pattern => sub { my ($req) = @_; ... })

=item route::add($method, $pattern, $handler)

Add a route (C<route::any> and C<route::add> with an undef method: any method). In C<$pattern>, C<:name> matches one path segment and C<*name> the rest of the path, both
URL-decoded into C<< $req->{params} >>. Routes are tried in the order they were added; a C<GET> route also
answers C<HEAD>. A path that matches a route of another method gets 405 with an C<Allow> header, no match 404.

=item route::list(), route::clear()

The table as C<[method, pattern]> pairs (C<*> for any method) / empty it.

=back

=head2 Requests

The handler gets a hash: C<method>, C<path> (as sent), C<query> (the raw query string), C<args> (the query
decoded: a key given more than once becomes an array reference), C<params> (from the pattern), C<headers>
(names lower-cased; repeated headers joined with C<, >), C<body> (the raw bytes), C<json> (the body decoded,
for a JSON content type; invalid JSON gets 400 before the handler runs), C<protocol> and C<remote> (nginx's
C<X-Real-IP> or C<X-Forwarded-For>).

=head2 Responses

A handler returns C<[$status, [@headers], $body]>, or something that becomes one: a hash or an array (of
other things than a response) becomes JSON, undef 204, anything else text. C<$body> is a string, an array of
strings or a code reference, which streams. Character strings are sent as UTF-8. A handler may also
C<die> with a response; any other C<die> gives a 500 (the message goes to STDERR, not to the client).

=over

=item json($data [, $status, @headers])

=item text($string [, $status, @headers])

=item stream($status, [@headers], sub { my ($write) = @_; ... })

C<< $write->($text) >> sends a piece at once and returns false once the client is gone. The response has no
length: closing the connection ends it. For C<text/event-stream>, C<X-Accel-Buffering: no> and
C<Cache-Control: no-cache> are added, so nginx passes events on as they come even with buffering on.

=item sse($data, %fields)

One server-sent event as text: C<$data> a string or a structure (sent as JSON), fields C<event>, C<id>,
C<retry>.

=back

=head2 Serving

=over

=item run(listen => $where, %options)

Serve until SIGINT or SIGTERM (after the current request). C<$where> is a socket path (an old socket file is
replaced, and removed at the end) or C<host:port>. Options: C<mode> (the socket file's permissions, e.g.
C<0660> for a group nginx is in), C<backlog> (128), C<timeout> (seconds to wait for a request's bytes, 30),
C<max_header> (64 KB), C<max_body> (16 MB; 413 above), C<log> (a line per request on STDERR), C<on_start>
(called with the listening socket). Request bodies come with a C<Content-Length>: nginx sends them so with
C<proxy_request_buffering on> (the default); a chunked request body gets 501.

=item dispatch(\%req)

The response to a request hash: what C<run> does after parsing, for tests without a socket.

=back

=head1 SEE ALSO

L<ROCm::Inference> (whose OpenAI-compatible server uses it), L<IO::Socket::UNIX>, L<JSON::PP>.

=head1 AUTHOR

Tim Aerts <aardbeiplantje@gmail.com>

=head1 LICENSE

The Unlicense: public domain. See L<https://unlicense.org/>.

=cut
