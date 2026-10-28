use strict;
use warnings;
use Test::More;

use File::Temp qw(tempdir);
use IO::Socket::UNIX;
use Socket qw(SOCK_STREAM);
use JSON::PP ();
use MIME::Base64 qw(decode_base64);
use Time::HiRes qw(sleep);

use ROCm::Inference qw(:all);

# Embedding models: embed, and rocm-inference-server's /v1/embeddings (an embedding model alone, and next to a chat
# model when ROCM_INFERENCE_TEST_MODEL is set too). Needs a GPU and ROCM_INFERENCE_TEST_EMBED_MODEL=<gguf>
# (EmbeddingGemma).
my $path = $ENV{ROCM_INFERENCE_TEST_EMBED_MODEL};
plan skip_all => 'set ROCM_INFERENCE_TEST_EMBED_MODEL to an embedding GGUF model to run' unless $path && -f $path;

sub cosine {
    my ($a, $b) = @_;
    my ($ab, $aa, $bb) = (0, 0, 0);
    for my $i (0 .. $#$a) { $ab += $a->[$i] * $b->[$i]; $aa += $a->[$i]**2; $bb += $b->[$i]**2 }
    return $ab / sqrt($aa * $bb);
}

my $m = load_model($path);
kv_cache_init($m, 2048);
my $d = embedding_dim($m);
ok($d > 0, "embedding_dim: $d");
my @e = map { embed($m, tokenize($m, $_)) } "def add(a, b):\n    return a + b", "def sum_two(x, y):\n    return x + y",
    'The weather in Ghent is rainy today.';
is(scalar @{ $e[0] }, $d, 'embed: that many values');
my $len = 0;
$len += $_ * $_ for @{ $e[0] };
ok(abs($len - 1) < 1e-4, 'normalized');
ok(cosine(@e[0, 1]) > cosine(@e[0, 2]) + 0.1, sprintf('similar code closer (%.3f) than a weather sentence (%.3f)',
    cosine(@e[0, 1]), cosine(@e[0, 2])));
my $raw = embed($m, tokenize($m, 'x'), 0);
my $rl = 0;
$rl += $_ * $_ for @$raw;
ok(abs($rl - 1) > 1e-3, 'normalize 0: as pooled');
ok(!eval { embed($m, [ (2) x 3000 ]); 1 }, 'an input longer than the context dies');
free_model($m);

plan skip_all => 'Web::Route is not installed' unless eval { require Web::Route; 1 };
my $dir = tempdir(CLEANUP => 1);
my $sock = "$dir/e.sock";
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
    return ($status, $J->decode($rest));
}
sub server {
    my @args = @_;
    my $pid = open my $srv, '-|', $^X, (map { "-I$_" } @INC), 'bin/rocm-inference-server', '--listen', $sock,
        '--nolog', @args;
    die "cannot start the server: $!" unless $pid;
    for (1 .. 1200) { last if -S $sock; sleep 0.1 }
    return ($pid, $srv);
}

SKIP: {
    skip "Minijinja $ROCm::Inference::MINIJINJA_VERSION is not installed", 1 unless ROCm::Inference::_require_minijinja();
    my ($pid, $srv) = server('--model', $path, '--name', 'emb');
    ok(-S $sock, 'an embedding model alone: the server listens');
    my ($st, $r) = http(POST => '/v1/embeddings', { input => "def add(a, b):\n    return a + b", model => 'emb' });
    is($st, 200, 'a string');
    is($r->{model}, 'emb', 'the model');
    ok(cosine($r->{data}[0]{embedding}, $e[0]) > 0.99999, 'its embedding = embed');
    ok($r->{usage}{prompt_tokens} > 0, 'usage');
    ($st, $r) = http(POST => '/v1/embeddings', { input => [ 'a', 'b', 'c' ] });
    is_deeply([ map { $_->{index} } @{ $r->{data} } ], [ 0, 1, 2 ], 'an array of strings: three, in order');
    my $ids = ROCm::Inference::tokenize(load_model($path), 'The weather in Ghent is rainy today.');
    ($st, $r) = http(POST => '/v1/embeddings', { input => [ $ids, $ids ] });
    ok($st == 200 && cosine($r->{data}[1]{embedding}, $e[2]) > 0.99999, 'token arrays');
    ($st, $r) = http(POST => '/v1/embeddings', { input => $ids, encoding_format => 'base64' });
    my @b = unpack 'f<*', decode_base64($r->{data}[0]{embedding});
    ok(@b == $d && cosine(\@b, $e[2]) > 0.99999, 'base64');
    ($st, $r) = http(POST => '/v1/embeddings', { input => $ids, dimensions => 256 });
    my $v = $r->{data}[0]{embedding};
    my $n = 0;
    $n += $_ * $_ for @$v;
    ok(@$v == 256 && abs($n - 1) < 1e-4 && cosine($v, [ @{ $e[2] }[ 0 .. 255 ] ]) > 0.99999, 'dimensions 256: renormalized');
    ($st, $r) = http(POST => '/v1/embeddings', { input => 'word ' x 3000 });
    is($st, 400, 'too long: 400');
    like($r->{error}{message}, qr/more than the context/, '  with why');
    ($st, $r) = http(POST => '/v1/chat/completions', { messages => [ { role => 'user', content => 'hi' } ] });
    is($st, 400, 'no chat with an embedding model');
    kill TERM => $pid;
    close $srv;

    my $chat = $ENV{ROCM_INFERENCE_TEST_MODEL};
    skip 'set ROCM_INFERENCE_TEST_MODEL too for a chat model with an embedding model', 1 unless $chat && -f $chat;
    ($pid, $srv) = server('--model', $chat, '--name', 'chat', '--ctx', 4096, '--embedding-model', $path,
                          '--embedding-name', 'emb');
    ($st, $r) = http(GET => '/v1/models');
    is_deeply([ map { $_->{id} } @{ $r->{data} } ], [ 'chat', 'emb' ], 'with a chat model: both listed');
    ($st, $r) = http(POST => '/v1/embeddings', { input => "def add(a, b):\n    return a + b" });
    ok($st == 200 && cosine($r->{data}[0]{embedding}, $e[0]) > 0.99999, 'embeddings');
    ($st, $r) = http(POST => '/v1/chat/completions', { max_tokens => 8, temperature => 0,
        messages => [ { role => 'user', content => 'Say hi.' } ] });
    is($st, 200, 'and chat');
    kill TERM => $pid;
    close $srv;
}

done_testing;
