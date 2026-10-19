use strict;
use warnings;
use Test::More;
use Time::HiRes qw(time);

use ROCm::Inference qw(:all);

# Speculative decoding with an MTP draft head. Needs the GPU, Qwen3.8-Flash-Next and its head:
#   ROCM_INFERENCE_TEST_QWEN38=<.../Qwen3.8-Flash-Next-...-00001-of-00003.gguf>
#   ROCM_INFERENCE_TEST_DRAFT=<.../mtp-Qwen3.8-Flash-Next-Q4_0_ROCMFP4_FAST.gguf>
# (and ROCM_INFERENCE_TEST_MODEL, any other model, for the error path)
my ($q38, $draft) = @ENV{qw(ROCM_INFERENCE_TEST_QWEN38 ROCM_INFERENCE_TEST_DRAFT)};
plan skip_all => 'set ROCM_INFERENCE_TEST_QWEN38 and ROCM_INFERENCE_TEST_DRAFT to run'
    unless $q38 && -f $q38 && $draft && -f $draft;

if (my $other = $ENV{ROCM_INFERENCE_TEST_MODEL}) {
    my $o = load_model($other);
    my $err = eval { set_draft_model($o, $draft); 1 } ? '' : $@;
    like($err, qr/set_draft_model/, 'a head for another model dies');
}

my $m = load_model($q38);
set_draft_model($m, $draft);  # before kv_cache_init
kv_cache_init($m, 4096);
my $prompt = tokenize($m, "<|im_start|>user\nWrite a Perl function that reverses the words of a sentence."
    . "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");

my $err = eval { set_draft_tokens($m, 4); 1 } ? '' : $@;
like($err, qr/set_draft_tokens/, 'more than 3 drafts per pass dies');

# The same tokens and text with and without the head (1 and 3 drafts per pass), greedy and sampled
for my $temp (0, 0.8) {
    my (%ids, %text, %secs);
    for my $with (3, 1, 0) {
        set_draft_model($m, $with ? $draft : undef);
        set_draft_tokens($m, $with) if $with;
        reset($m);
        my $t = '';
        my $t0 = time;
        my ($ids) = generate($m, $prompt, {
            max_new_tokens => 160, temperature => $temp, seed => 11,
            on_token       => sub { $t .= $_[1]; 0 },
        });
        ($ids{$with}, $text{$with}, $secs{$with}) = ($ids, $t, time - $t0);
    }
    for my $k (1, 3) {
        is_deeply($ids{$k}, $ids{0}, "temperature $temp, $k drafts per pass: the same "
            . scalar(@{ $ids{0} }) . ' tokens with the head');
        is($text{$k}, $text{0}, "temperature $temp, $k drafts per pass: the same streamed text");
    }
    note(sprintf('temperature %.1f: %.1f / %.1f tok/s with the head (1 / 3 drafts), %.1f without', $temp,
                 @{ $ids{1} } / $secs{1}, @{ $ids{3} } / $secs{3}, @{ $ids{0} } / $secs{0}));
}

# A turn that ends by itself (end of turn), then a second one continuing the sequence
set_draft_model($m, $draft);
set_draft_tokens($m, 2);
reset($m);
my ($ids, $why) = generate($m, $prompt, { max_new_tokens => 1000, temperature => 0 });
is($why, STOP_EOG, 'the reply ends at the end of the turn (' . scalar(@$ids) . ' tokens)');
is(n_past($m), @$prompt + @$ids, 'n_past counts the prompt and the reply (no draft left decoded)');
like(detokenize($m, $ids), qr/sub\s+\w+/, 'the reply has a Perl sub');

done_testing;
