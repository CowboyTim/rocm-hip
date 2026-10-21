use strict;
use warnings;
use utf8;
use Test::More;

use ROCm::Inference qw(:all);

# Needs a model and the Minijinja module (chat templates are rendered in Perl)
my $path = $ENV{ROCM_INFERENCE_TEST_MODEL};
plan skip_all => 'set ROCM_INFERENCE_TEST_MODEL to a GGUF model to run' unless $path && -f $path;
plan skip_all => "Minijinja $ROCm::Inference::MINIJINJA_VERSION is not installed" unless ROCm::Inference::_require_minijinja();

binmode Test::More->builder->$_, ':encoding(UTF-8)' for qw(output failure_output todo_output);
sub short { my $t = shift // ''; $t =~ s/\s+/ /g; length $t > 80 ? substr($t, 0, 77) . '...' : $t }

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

# Thinking on, streamed by kind, with a thinking budget: the reasoning is cut short, then the answer follows
my $q = [{ role => 'user', content => 'Is 91 a prime number? Answer in one sentence.' }];
my (%deltas, @kinds);
my ($content, $why3, $msg) = chat($m, $q, { temperature => 0, max_new_tokens => 200, max_thinking_tokens => 16,
    vars => { enable_thinking => 1 },
    on_delta => sub { $deltas{ $_[1] } .= $_[0]; push @kinds, $_[1] unless @kinds && $kinds[-1] eq $_[1]; 0 } });
is($why3, STOP_EOG, 'thinking budget: the answer ends at end-of-turn');
is("@kinds", 'reasoning content', 'streamed: reasoning, then content');
ok(length $msg->{reasoning_content}, "reasoning: " . short($msg->{reasoning_content}));
unlike($content, qr/<\/?think>/, "content without tags: " . short($content));
is($deltas{reasoning}, $msg->{reasoning_content}, 'the reasoning deltas add up to reasoning_content');
is($deltas{content}, $content, 'the content deltas add up to the content');

# A tool call round trip: the call is parsed, the result goes back, and the next turn reuses the first
my @tools = ({ type => 'function', function => { name => 'get_weather', description => 'Current weather for a city',
    parameters => { type => 'object', properties => { city => { type => 'string' } }, required => ['city'] } } });
my @chat = ({ role => 'user', content => 'What is the weather in Ghent right now?' });
my (undef, undef, $call) = chat($m, \@chat, { temperature => 0, max_new_tokens => 300, vars => { tools => \@tools } });
is($call->{tool_calls}[0]{function}{name}, 'get_weather', 'the reply calls get_weather');
like($call->{tool_calls}[0]{function}{arguments}{city} // '', qr/Gh?ent/i, 'with the city');
push @chat, $call, { role => 'tool', content => '{"temperature_c": 14, "sky": "light rain"}' };
my $answer = chat($m, \@chat, { temperature => 0, max_new_tokens => 300, vars => { tools => \@tools } });
my ($reused3, $prefilled3) = last_prompt_stats($m);
like($answer, qr/14/, "the answer uses the tool's result: " . short($answer));
ok($prefilled3 < 64, "the tool turn reused $reused3 tokens, prefilled $prefilled3");

done_testing;
