use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir);

use ROCm::Inference qw(:all);

# The KV block store from Perl. Needs a GPU and a model: ROCM_INFERENCE_TEST_MODEL=<gguf>
my $path = $ENV{ROCM_INFERENCE_TEST_MODEL};
plan skip_all => 'set ROCM_INFERENCE_TEST_MODEL to a GGUF model to run' unless $path && -f $path;
my $dir = tempdir('rocm-inference-store-XXXXXX', DIR => (-d '/dev/shm' ? '/dev/shm' : undef), CLEANUP => 1);

my $m = load_model($path);
# Gemma-4's sliding-window layers keep a ring of their last positions, which a block store cannot hold: one
# cache of every layer's history instead (no effect on other models)
local $ENV{INFERENCE_SWA_FULL} = 1;
kv_cache_init($m, 4096);
store_open($m, $dir, 2);
my $st = store_stats($m);
ok($st->{block_tokens} > 0 && $st->{block_bytes} > 0, "store open: $st->{block_tokens}-token blocks of $st->{block_bytes} bytes");

my $text = join ' ', map { "Line $_ of a long system prompt that many requests share." } 1 .. 120;
my $prompt = tokenize($m, $text);
my $nb = int((@$prompt - 1) / $st->{block_tokens});
ok($nb >= 2, scalar(@$prompt) . " prompt tokens, $nb full blocks");

my ($a1) = generate($m, $prompt, { max_new_tokens => 8, temperature => 0, reuse_prefix => 1 });
$st = store_stats($m);
ok($st->{blocks_published} >= $nb, "first request publishes $st->{blocks_published} blocks");
my @names = store_block_names($m, $prompt);
is(scalar @names, int(@$prompt / $st->{block_tokens}), 'store_block_names: one name per full block');
ok(!grep({ !-f "$dir/$_" } @names), 'and those files exist in the store');

# A new sequence with the same prompt starts from the store
reset($m);
my ($a2) = generate($m, $prompt, { max_new_tokens => 8, temperature => 0, reuse_prefix => 1 });
my ($reused, $prefilled) = last_prompt_stats($m);
$st = store_stats($m);
ok($reused >= $st->{block_tokens}, "second request: $reused tokens from the store, $prefilled prefilled");
is($a2->[0], $a1->[0], 'same first token');

# Eviction is deleting files: the next request prefills again and republishes
unlink "$dir/$_" for @names;
reset($m);
generate($m, $prompt, { max_new_tokens => 4, temperature => 0, reuse_prefix => 1 });
($reused) = last_prompt_stats($m);
is($reused, 0, 'after evicting the blocks nothing is reused');
ok(-f "$dir/$names[0]", 'and the blocks are published again');

store_close($m);
done_testing;
