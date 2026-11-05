use strict;
use warnings;
use Test::More;

use File::Temp qw(tempdir);
use IO::Socket::UNIX;
use Socket qw(SOCK_STREAM);
use JSON::PP ();
use Time::HiRes qw(sleep);

# bin/rocm-inference-server --models-dir: the directory's models listed, each loaded when a request names it, the
# least recently used unloaded for it (--models-max 1), /models/load and /models/unload; once as is and once with
# --parallel 2. The directory: the test model under two names (two models to switch between), the embedding model
# if given, and files that are not models (a later shard, a non-GGUF file).
my $path = $ENV{ROCM_INFERENCE_TEST_MODEL};
my $emb = $ENV{ROCM_INFERENCE_TEST_EMBED_MODEL};
plan skip_all => 'set ROCM_INFERENCE_TEST_MODEL to a GGUF model to run' unless $path && -f $path;
plan skip_all => 'Web::Route is not installed' unless eval { require Web::Route; 1 };
require ROCm::Inference;
plan skip_all => "Minijinja $ROCm::Inference::MINIJINJA_VERSION is not installed"
    unless ROCm::Inference::_require_minijinja();

my $dir = tempdir(CLEANUP => 1);
my $models = "$dir/models";
mkdir $models or die;
symlink $path, "$models/alpha.gguf" or die;
symlink $path, "$models/beta.gguf" or die;
symlink $path, "$models/big-00002-of-00003.gguf" or die;  # not a first shard: left out
{ open my $fh, '>', "$models/notes.gguf" or die; print $fh "not a GGUF\n" }
symlink $emb, "$models/embed.gguf" or die if $emb && -f $emb;
my $sock = "$dir/llm.sock";
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
    return ($status, $rest);
}

sub models {
    my ($st, $body) = http(GET => '/v1/models');
    return { map { $_->{id} => $_ } @{ $J->decode($body)->{data} } };
}

sub chat {
    my ($model, %x) = @_;
    my ($st, $body) = http(POST => '/v1/chat/completions', { temperature => 0, max_tokens => 16,
        chat_template_kwargs => { enable_thinking => JSON::PP::false }, defined $model ? (model => $model) : (),
        messages => [ { role => 'user', content => 'Name a primary color.' } ], %x });
    return ($st, $st == 200 ? $J->decode($body) : $body);
}

for my $parallel (1, 2) {
    note("--parallel $parallel");
    unlink $sock;
    my $pid = open my $srv, '-|', $^X, (map { "-I$_" } @INC), 'bin/rocm-inference-server', '--models-dir', $models,
        '--models-max', 1, '--listen', $sock, '--ctx', 4096, '--nolog', '--parallel', $parallel;
    die "cannot start the server: $!" unless $pid;
    for (1 .. 600) { last if -S $sock; sleep 0.1 }
    ok(-S $sock, 'the server listens') or BAIL_OUT('no server');

    my $m = models();
    is_deeply([ sort grep { $_ ne 'embed' } keys %$m ], [qw(alpha beta)], 'the models: not a later shard, not a non-GGUF');
    is($m->{alpha}{status}{value}, 'unloaded', 'nothing loaded at start');
    is($m->{alpha}{kind}, 'chat', 'its kind');
    is($m->{alpha}{context_length}, 4096, 'the context it gets (--ctx)');
    is($m->{embed}{kind}, 'embedding', 'the embedding model') if $m->{embed};

    my ($st, $r) = chat('alpha');
    is($st, 200, 'a chat with alpha');
    is($r->{model}, 'alpha', 'answered by alpha');
    like($r->{choices}[0]{message}{content}, qr/\w/, 'an answer');
    is(models()->{alpha}{status}{value}, 'loaded', 'alpha loaded');

    ($st, $r) = chat('beta', stream => JSON::PP::false);
    is($st, 200, 'a chat with beta');
    is($r->{model}, 'beta', 'answered by beta');
    $m = models();
    is($m->{beta}{status}{value}, 'loaded', 'beta loaded');
    is($m->{alpha}{status}{value}, 'unloaded', '--models-max 1: alpha unloaded for it');

    ($st, $r) = chat(undef);
    is($st, 200, 'a chat that names no model');
    is($r->{model}, 'beta', 'gets the last one used');

    ($st) = chat('gamma');
    is($st, 404, 'an unknown model: 404');

    ($st, $r) = http(POST => '/models/unload', { model => 'beta' });
    is($st, 200, 'POST /models/unload');
    is(models()->{beta}{status}{value}, 'unloaded', 'beta unloaded');
    ($st, $r) = http(POST => '/models/load', { model => 'alpha' });
    is($st, 200, 'POST /models/load');
    is(models()->{alpha}{status}{value}, 'loaded', 'alpha loaded');

    if ($m->{embed}) {
        ($st, $r) = http(POST => '/v1/embeddings', { model => 'embed', input => 'hello' });
        is($st, 200, 'embeddings from the embedding model');
        is(scalar @{ $J->decode($r)->{data}[0]{embedding} }, 768, 'its vector');
        ($st) = chat('embed');
        is($st, 400, 'a chat with the embedding model: 400');
        ($st) = http(POST => '/v1/embeddings', { model => 'alpha', input => 'hello' });
        is($st, 400, 'embeddings from a chat model: 400');
    }

    symlink $path, "$models/delta.gguf" or die;
    ok(models()->{delta}, 'a model added to the directory is listed');
    unlink "$models/delta.gguf";
    ok(!models()->{delta}, 'and gone when removed');

    kill TERM => $pid;
    close $srv;
}

done_testing;
