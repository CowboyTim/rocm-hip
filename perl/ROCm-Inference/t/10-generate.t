use strict;
use warnings;
use Test::More;

use ROCm::Inference qw(:all);

# Needs a GPU and a model: ROCM_INFERENCE_TEST_MODEL=/dev/shm/models/Qwen3.5-4B-ROCMFP4.gguf
my $path = $ENV{ROCM_INFERENCE_TEST_MODEL};
plan skip_all => 'set ROCM_INFERENCE_TEST_MODEL to a GGUF model to run' unless $path && -f $path;

my $m = load_model($path);
is(set_prefill_batch($m, 512), 0, 'set_prefill_batch before kv_cache_init');
kv_cache_init($m, 2048);
is(n_ctx($m), 2048, 'n_ctx');
my $err = eval { set_prefill_batch($m, 1024); 1 } ? '' : $@;
like($err, qr/set_prefill_batch/, 'set_prefill_batch after kv_cache_init dies');
ok(vocab_size($m) > 1000, 'vocab_size');
ok(token_is_eog($m, eos_id($m)), 'EOS ends generation');

my $ids = tokenize($m, 'The capital of France is');
ok(@$ids > 3, 'tokenize');
# (models that want a BOS, Gemma, get one added: it detokenizes as <bos>)
sub text_ids { my $t = tokenize($m, shift); shift @$t if @$t && $t->[0] == bos_id($m); $t }
is(detokenize($m, text_ids('The capital of France is')), 'The capital of France is', 'detokenize round trip');
my $u = "Gr\x{fc}\x{df}e aus K\x{f6}ln \x{2603}";
is(detokenize($m, text_ids($u)), $u, 'non-ASCII text round trip (character strings)');

# Greedy, streamed: the callback text adds up to the detokenized tokens (an instruction-tuned model may end
# its turn early on a plain prompt: ignore_eog)
my $streamed = '';
my ($out, $why) = generate($m, $ids, {
    max_new_tokens => 12, temperature => 0, ignore_eog => 1,
    on_token => sub { $streamed .= $_[1]; 0 },
});
is(scalar @$out, 12, '12 tokens');
is($why, STOP_MAX_TOKENS, 'stop reason: max tokens');
is($streamed, detokenize($m, $out), 'streamed text = detokenized tokens');
like($streamed, qr/Paris/, 'greedy continuation mentions Paris');
is(n_past($m), @$ids + 12, 'n_past counts the prompt and the emitted tokens');

# Repeatable from a reset; a stop string; the callback stops it; a die in the callback propagates
reset($m);
my ($again) = generate($m, $ids, { max_new_tokens => 12, temperature => 0, ignore_eog => 1 });
is_deeply($again, $out, 'greedy is repeatable after reset');
reset($m);
my $text = '';
(undef, $why) = generate($m, $ids, { max_new_tokens => 12, temperature => 0, stop => ['Paris'],
                                     on_token => sub { $text .= $_[1]; 0 } });
is($why, STOP_STRING, 'stop string');
unlike($text, qr/Paris/, 'the stop string is not delivered');
reset($m);
my $calls = 0;
(undef, $why) = generate($m, $ids, { max_new_tokens => 12, temperature => 0, on_token => sub { ++$calls >= 3 } });
is($why, STOP_CALLBACK, 'the callback stops it');
is($calls, 3, 'after 3 tokens');
reset($m);
my $died = eval { generate($m, $ids, { max_new_tokens => 4, on_token => sub { die "boom\n" } }); 1 } ? '' : $@;
is($died, "boom\n", 'a die in on_token propagates');

# Hand-rendered chat turns with prefix reuse
my $sys = "<|im_start|>system\nYou are a helpful assistant. Answer briefly.<|im_end|>\n";
my $turn = "$sys<|im_start|>user\nName three primary colors.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
reset($m);
my ($a1, $r1) = generate($m, tokenize($m, $turn), { max_new_tokens => 64, temperature => 0, reuse_prefix => 1 });
is($r1, STOP_EOG, 'a chat turn ends at end-of-turn');
my $reply = detokenize($m, $a1);
my $turn2 = "$sys<|im_start|>user\nName three primary colors.<|im_end|>\n<|im_start|>assistant\n$reply<|im_end|>\n"
          . "<|im_start|>user\nAnd three secondary colors?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
my ($a2) = generate($m, tokenize($m, $turn2), { max_new_tokens => 64, temperature => 0, reuse_prefix => 1 });
my ($reused, $prefilled) = last_prompt_stats($m);
ok($reused > 0, "turn 2 reused $reused tokens, prefilled $prefilled");
like(detokenize($m, $a2), qr/green|orange|purple/i, 'turn 2 answers about secondary colors');

my $logits = get_logits($m);
is(length($logits), 4 * vocab_size($m), 'get_logits: packed floats');

free_model($m);
my $freed = eval { n_ctx($m); 1 } ? '' : $@;
like($freed, qr/already freed/, 'a freed handle dies');

done_testing;
