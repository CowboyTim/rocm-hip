use strict;
use warnings;
use utf8;
use Test::More;

use ROCm::Inference;

# Splitting a reply into reasoning, content and tool calls as it streams (chat), without a model: the result
# must not depend on how the text is cut into pieces, and the deltas of a kind add up to the message's field
binmode Test::More->builder->$_, ':encoding(UTF-8)' for qw(output failure_output todo_output);

sub split_reply {
    my ($start, @pieces) = @_;
    my %deltas;
    my $s = ROCm::Inference::_reply_splitter($start, sub { $deltas{ $_[1] } .= $_[0]; 0 });
    $s->{feed}->($_) for @pieces;
    return ($s->{finish}->(), \%deltas);
}

# every way of cutting must give the same message
sub check {
    my ($name, $start, $text, $want) = @_;
    my @cuts = ([$text], [split //, $text]);
    srand(7);
    for (1 .. 20) {
        my ($t, @p) = ($text);
        while (length $t) { push @p, substr($t, 0, 1 + int(rand(9)), '') }
        push @cuts, \@p;
    }
    for my $p (@cuts) {
        my ($msg, $deltas) = split_reply($start, @$p);
        unless (is_deeply($msg, $want, "$name (" . @$p . ' pieces)')) { diag explain $msg; return }
        for my $k (qw(reasoning content)) {
            my $field = $k eq 'reasoning' ? 'reasoning_content' : 'content';
            is($deltas->{$k} // '', $msg->{$field} // '', "$name: the $k deltas add up") or return;
        }
    }
}

check('thinking from the generation prompt', 'reasoning', "The user wants colors.\n</think>\n\nRed, blue, yellow.",
      { role => 'assistant', reasoning_content => 'The user wants colors.', content => 'Red, blue, yellow.' });
check('a <think> block the model opens', 'content', "<think>\nHmm, a  test.\n\n</think>\n\nAnswer:\n\n42\n",
      { role => 'assistant', reasoning_content => 'Hmm, a  test.', content => "Answer:\n\n42" });
check('no thinking', 'content', "\n\nJust text, < and </think-ish> and <tool_ stuff.",
      { role => 'assistant', content => 'Just text, < and </think-ish> and <tool_ stuff.' });
check('an unfinished reasoning', 'reasoning', "Step 1 …\nStep 2",
      { role => 'assistant', reasoning_content => "Step 1 …\nStep 2", content => '' });
check('tool call, Qwen3.5 XML', 'reasoning',
      "Need the weather.\n</think>\n\nLet me check.\n\n<tool_call>\n<function=weather>\n<parameter=city>\nGent\n"
      . "</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>",
      { role => 'assistant', reasoning_content => 'Need the weather.', content => 'Let me check.',
        tool_calls => [ { type => 'function', function => { name => 'weather',
                                                             arguments => { city => 'Gent', days => '2' } } } ] });
check('tool calls, Qwen3 JSON', 'content',
      qq[<tool_call>\n{"name": "weather", "arguments": {"city": "中文"}}\n</tool_call>\n<tool_call>\n]
      . qq[{"name": "time", "arguments": "{\\"tz\\": \\"CET\\"}"}\n</tool_call>],
      { role => 'assistant', content => '',
        tool_calls => [ { type => 'function', function => { name => 'weather', arguments => { city => '中文' } } },
                        { type => 'function', function => { name => 'time', arguments => { tz => 'CET' } } } ] });
check('a tool call that does not parse', 'content', "Ok <tool_call>\nnot json\n</tool_call>",
      { role => 'assistant', content => 'Ok', invalid_tool_calls => ["\nnot json\n"] });
check('multi-line value', 'content',
      "<tool_call>\n<function=write>\n<parameter=text>\nline 1\n\nline 2\n</parameter>\n</function>\n</tool_call>",
      { role => 'assistant', content => '',
        tool_calls => [ { type => 'function', function => { name => 'write',
                                                             arguments => { text => "line 1\n\nline 2" } } } ] });

# on_delta asking to stop is reported by feed
my $s = ROCm::Inference::_reply_splitter('content', sub { $_[0] =~ /stop/ });
ok(!$s->{feed}->('go on '), 'feed: no stop');
ok($s->{feed}->('now stop'), 'feed: on_delta asked to stop');
is($s->{kind}->(), 'content', 'kind');

# the message renders back with a Qwen-style template as the model wrote it
SKIP: {
    skip 'Minijinja 0.1.1 is not installed', 1 unless ROCm::Inference::_require_minijinja();
    my ($msg) = split_reply('reasoning', "Why.\n</think>\n\nBecause.");
    my $tmpl = q[{%- for m in messages %}{{ m.role }}: {% if m.reasoning_content %}<think>
{{ m.reasoning_content }}
</think>

{% endif %}{{ m.content }}{% endfor %}];
    is(ROCm::Inference::_render_template($tmpl, { messages => [$msg] }),
       "assistant: <think>\nWhy.\n</think>\n\nBecause.", 'renders back');
}

done_testing;
