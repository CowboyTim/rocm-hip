use strict;
use warnings;
use Test::More;

use ROCm::Inference qw(:all);

# The engine's events (set_event_handler): the handler gets (name, { seq, n0, n1, ms }) on a real generate, the
# generation is the same with and without it, and its return on prefill_progress (true, or a die) cancels the prefill.
# Needs a GPU and a model: ROCM_INFERENCE_TEST_MODEL=/dev/shm/models/Qwen3.5-4B-ROCMFP4.gguf
my $path = $ENV{ROCM_INFERENCE_TEST_MODEL};
plan skip_all => 'set ROCM_INFERENCE_TEST_MODEL to a GGUF model to run' unless $path && -f $path;

my $text = join '', map { 'The quick brown fox jumps over the lazy dog near the river bank. ' } 1 .. 100;
my $opts = { max_new_tokens => 6, temperature => 0, ignore_eog => 1, reuse_prefix => 1 };

my $m = load_model($path);
kv_cache_init($m, 4096);
my @ev;
set_event_handler($m, sub { push @ev, [@_]; 0 });  # (0: a true return would cancel the prefill)
my $ids = tokenize($m, $text . 'Question: what color is the fox?');
my ($with) = generate($m, $ids, $opts);
my @names = map { $_->[0] } @ev;
is($names[0], 'prefill_begin', 'prefill_begin comes first');
is(scalar(grep { $_ eq 'prefill_begin' } @names), 1, 'one prefill_begin per generation');
is(scalar(grep { $_ eq 'prefill_end' } @names), 1, 'one prefill_end per generation');
my ($end) = grep { $_->[0] eq 'prefill_end' } @ev;
is(ref $end->[1], 'HASH', 'the info is a hash');
ok(exists $end->[1]{$_}, "info has $_") for qw(seq n0 n1 ms);
ok($end->[1]{n0} > 0, 'prefill_end n0: the tokens prefilled');
cmp_ok(scalar(grep { $_->[0] eq 'prefill_progress' } @ev), '>', 0, 'a long prompt reports prefill_progress');

# the same generation with no handler: the same tokens
set_event_handler($m, undef);
my $n_before = @ev;
my ($without) = generate($m, $ids, $opts);
is(scalar(@ev), $n_before, 'no events once the handler is removed');
is_deeply($without, $with, 'the same tokens with and without the handler');

# the handler's cancel. Each on a fresh model: the same prompt again would be reused whole, with no prefill steps
{
    my $c = load_model($path);
    kv_cache_init($c, 4096);
    set_event_handler($c, sub { $_[0] eq 'prefill_progress' ? 1 : 0 });
    my $ok = eval { generate($c, tokenize($c, $text . 'Question: what color is the fox?'), $opts); 1 };
    ok(!$ok, 'a true return on prefill_progress cancels the prefill');
    like($@, qr/cancelled/, 'the cancel is the error');
}
{
    my $c = load_model($path);
    kv_cache_init($c, 4096);
    my @warns;
    local $SIG{__WARN__} = sub { push @warns, @_ };
    set_event_handler($c, sub { die "boom\n" if $_[0] eq 'prefill_progress'; 0 });
    my $ok = eval { generate($c, tokenize($c, $text . 'Question: what color is the fox?'), $opts); 1 };
    ok(!$ok, 'a die on prefill_progress cancels the prefill');
    ok(scalar(grep { /event handler died, the work is cancelled: boom/ } @warns), 'the die is a warning too');
}
{
    my $c = load_model($path);
    kv_cache_init($c, 4096);
    set_event_handler($c, sub { $_[0] ne 'prefill_progress' });
    my ($ignored) = generate($c, tokenize($c, $text . 'Question: what color is the fox?'), $opts);
    is_deeply($ignored, $with, 'true returns on the other events change nothing');
}
done_testing;
