use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir);

use ROCm::Inference qw(:all);

# Staging (load_model's stage): the GPU tensors copied to a tmpfs and mapped from there give the same output as the
# GGUF mapped directly; the stage file is reused by the next load; without room the model loads unstaged.
# Needs a GPU, a model and a tmpfs with room for it: ROCM_INFERENCE_TEST_MODEL=<gguf> (stage under /dev/shm, or
# ROCM_INFERENCE_TEST_STAGE=<dir>)
my $path = $ENV{ROCM_INFERENCE_TEST_MODEL};
plan skip_all => 'set ROCM_INFERENCE_TEST_MODEL to a GGUF model to run' unless $path && -f $path;
my $base = $ENV{ROCM_INFERENCE_TEST_STAGE} // '/dev/shm';
plan skip_all => "no stage directory $base" unless -d $base && -w $base;
my $dir = tempdir('rocm-inference-stage-XXXXXX', DIR => $base, CLEANUP => 1);

my $prompt = 'The capital of France is Paris. The capital of Germany is';
sub run {
    my ($m) = @_;
    kv_cache_init($m, 1024);
    my ($ids) = generate($m, tokenize($m, $prompt), { max_new_tokens => 12, temperature => 0, ignore_eog => 1 });
    return "@$ids";
}

my $m = load_model($path);
is(stage_info($m), '', 'no staging asked: no stage info');
my $direct = run($m);
undef $m;

$m = load_model($path, { stage => $dir });
like(stage_info($m), qr/^staged \Q$dir\E\/rocm-inference-stage-[0-9a-f]{32}\.bin \(\d+ MB\)\z/, 'staged: ' . stage_info($m));
is(run($m), $direct, 'staged: the same tokens as mapped directly');
undef $m;
my @files = glob("$dir/rocm-inference-stage-*.bin");
is(scalar @files, 1, 'one stage file, kept after the model is freed');

$m = load_model($path, { stage => $dir });
like(stage_info($m), qr/^reused \Q$files[0]\E /, 'the next load reuses it');
is(run($m), $direct, 'reused: the same tokens');
undef $m;

$m = load_model($path, { stage => "$dir/missing" });
like(stage_info($m), qr/^not staged: /, 'no such directory: not staged, ' . stage_info($m));
is(run($m), $direct, 'not staged: still runs, mapped directly');
undef $m;

ok(!eval { load_model($path, 'not a hash'); 1 }, 'options must be a hash reference');

done_testing;
