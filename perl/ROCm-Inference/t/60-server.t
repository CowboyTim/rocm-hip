use strict;
use warnings;
use Test::More;

use File::Temp qw(tempdir);
use IO::Socket::UNIX;
use Socket qw(SOCK_STREAM);
use JSON::PP ();
use Time::HiRes qw(sleep);

# bin/rocm-inference-server on a unix socket: models, a plain and a streamed chat, a tool call round trip
my $path = $ENV{ROCM_INFERENCE_TEST_MODEL};
plan skip_all => 'set ROCM_INFERENCE_TEST_MODEL to a GGUF model to run' unless $path && -f $path;
plan skip_all => 'Web::Route is not installed' unless eval { require Web::Route; 1 };
require ROCm::Inference;
plan skip_all => "Minijinja $ROCm::Inference::MINIJINJA_VERSION is not installed"
    unless ROCm::Inference::_require_minijinja();

my $dir = tempdir(CLEANUP => 1);
my $sock = "$dir/llm.sock";
my $pid = open my $srv, '-|', $^X, (map { "-I$_" } @INC), 'bin/rocm-inference-server', '--model', $path,
    '--listen', $sock, '--ctx', 4096, '--name', 'test-model', '--nolog';
die "cannot start the server: $!" unless $pid;
for (1 .. 600) { last if -S $sock; sleep 0.1 }
ok(-S $sock, 'the server listens') or BAIL_OUT('no server');

my $J = JSON::PP->new->utf8->canonical;
sub http {
    my ($method, $url, $data) = @_;
    my $c = IO::Socket::UNIX->new(Peer => $sock, Type => SOCK_STREAM) or die "connect: $!";
    my $body = defined $data ? $J->encode($data) : '';
    print $c "$method $url HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: "
             . length($body) . "\r\n\r\n$body";
    local $/;
    my ($head, $rest) = split /\r\n\r\n/, <$c>, 2;
    my ($status) = $head =~ m{^HTTP/1\.1 (\d+)};
    return ($status, $rest, $head);
}

my ($st, $body) = http(GET => '/v1/models');
is($st, 200, 'GET /v1/models');
is($J->decode($body)->{data}[0]{id}, 'test-model', 'the model id');

my %q = (temperature => 0, chat_template_kwargs => { enable_thinking => JSON::PP::false });
($st, $body) = http(POST => '/v1/chat/completions', { %q, max_tokens => 48,
    messages => [ { role => 'user', content => 'Name three primary colors.' } ] });
my $r = $J->decode($body);
is($st, 200, 'chat');
like($r->{choices}[0]{message}{content}, qr/red|blue|yellow/i, 'the answer');
ok($r->{usage}{prompt_tokens} > 0 && $r->{usage}{completion_tokens} > 0, 'usage');

my $head;
($st, $body, $head) = http(POST => '/v1/chat/completions', { temperature => 0, max_tokens => 200, stream => JSON::PP::true,
    max_thinking_tokens => 16, stream_options => { include_usage => JSON::PP::true },
    messages => [ { role => 'user', content => 'Is 91 prime? One sentence.' } ] });
like($head, qr{Content-Type: text/event-stream}, 'an event stream');
my @ev = map { /^data: (.*)/ ? $1 : () } split /\n\n/, $body;
is($ev[-1], '[DONE]', 'it ends with [DONE]');
my @chunks = map { $J->decode($_) } @ev[ 0 .. $#ev - 1 ];
my ($reasoning, $content, $finish, $usage) = ('', '');
for (@chunks) {
    $usage = $_->{usage} if $_->{usage};
    my $c = $_->{choices}[0] or next;
    $reasoning .= $c->{delta}{reasoning_content} // '';
    $content .= $c->{delta}{content} // '';
    $finish = $c->{finish_reason} if $c->{finish_reason};
}
ok(length $reasoning, 'reasoning deltas');
like($content, qr/7|13|not/i, "content deltas: $content");
is($finish, 'stop', 'finish_reason');
ok($usage && $usage->{completion_tokens} > 0, 'a usage chunk');

my @tools = ({ type => 'function', function => { name => 'get_weather', description => 'Current weather for a city',
    parameters => { type => 'object', properties => { city => { type => 'string' } }, required => ['city'] } } });
my @msgs = ({ role => 'user', content => 'What is the weather in Ghent right now?' });
($st, $body) = http(POST => '/v1/chat/completions', { temperature => 0, max_tokens => 300, tools => \@tools, messages => \@msgs });
$r = $J->decode($body);
is($r->{choices}[0]{finish_reason}, 'tool_calls', 'finish_reason tool_calls');
my $call = $r->{choices}[0]{message}{tool_calls}[0];
is($call->{function}{name}, 'get_weather', 'the tool');
like($J->decode($call->{function}{arguments})->{city}, qr/Gh?ent/i, 'arguments as a JSON string');
push @msgs, $r->{choices}[0]{message},
    { role => 'tool', tool_call_id => $call->{id}, content => '{"temperature_c": 14, "sky": "light rain"}' };
($st, $body) = http(POST => '/v1/chat/completions', { temperature => 0, max_tokens => 300, tools => \@tools, messages => \@msgs });
like($J->decode($body)->{choices}[0]{message}{content}, qr/14/, "the answer uses the tool's result");

($st, $body) = http(POST => '/v1/chat/completions', { messages => [] });
is($st, 400, 'a bad request: 400');
like($J->decode($body)->{error}{message}, qr/messages/, 'with an OpenAI-style error');

kill TERM => $pid;
close $srv;
ok(!-e $sock, 'SIGTERM: the server stops and removes its socket');

done_testing;
