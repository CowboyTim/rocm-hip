use strict;
use warnings;
use utf8;
use Test::More;

use ROCm::Inference qw(:all);

# Needs a model and the Minijinja module (chat templates are rendered in Perl)
my $path = $ENV{ROCM_INFERENCE_TEST_MODEL};
plan skip_all => 'set ROCM_INFERENCE_TEST_MODEL to a GGUF model to run' unless $path && -f $path;
plan skip_all => "Minijinja $ROCm::Inference::MINIJINJA_VERSION is not installed" unless ROCm::Inference::_require_minijinja();

my $m = load_model($path);
kv_cache_init($m, 4096);
my @messages = ({ role => 'system', content => 'Answer briefly.' },
                { role => 'user', content => 'Name three primary colors.' });
my $prompt = chat_render($m, \@messages, enable_thinking => 0);
like($prompt, qr/<\|im_start\|>user\nName three primary colors\.<\|im_end\|>/, 'the template renders the user turn');
like($prompt, qr/<\|im_start\|>assistant\n<think>\n\n<\/think>\n\n\z/, 'generation prompt without thinking');
like(chat_render($m, [{ role => 'user', content => '你好, été' }]), qr/user\n你好, été<\|im_end\|>/,
     'non-ASCII text renders intact');

my ($reply, $why) = chat($m, \@messages, { temperature => 0, max_new_tokens => 48, vars => { enable_thinking => 0 } });
is($why, STOP_EOG, 'the reply ends at end-of-turn');
like($reply, qr/red|blue|yellow/i, "reply: $reply");

push @messages, { role => 'assistant', content => $reply }, { role => 'user', content => 'And three secondary colors?' };
my $reply2 = chat($m, \@messages, { temperature => 0, max_new_tokens => 48, vars => { enable_thinking => 0 } });
my ($reused) = last_prompt_stats($m);
ok($reused > 0, "turn 2 reused $reused tokens");
like($reply2, qr/green|orange|purple/i, "reply 2: $reply2");

done_testing;
