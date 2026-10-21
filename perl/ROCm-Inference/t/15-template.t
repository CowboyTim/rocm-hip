use strict;
use warnings;
use utf8;
use Test::More;

use ROCm::Inference;

# Chat template rendering without a model: the Python string and dict methods that templates call, tojson, the
# boolean template flags and UTF-8 text (the outputs match Python jinja2's)
plan skip_all => 'Minijinja is not installed' unless eval { require Minijinja; 1 };
binmode Test::More->builder->$_, ':encoding(UTF-8)' for qw(output failure_output todo_output);

sub r { ROCm::Inference::_render_template($_[0], $_[1] // {}) }

my $c = { content => "<think>\nwhy\n</think>\n\nanswer\n" };
is(r(q[{{ content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') }}], $c), 'why',
   'chained split/rstrip/lstrip with subscripts');
is(r(q[{{ content.split('</think>')[-1].lstrip('\n') }}], $c), "answer\n", 'the text after </think>');
is(r(q[{{ 'a</think>'.split('</think>')[-1] }}|]), '|', 'split keeps a trailing empty field');
is(r(q[{{ ' a  b '.split()|join(',') }}]), 'a,b', 'split on whitespace');
is(r(q[{{ (m.split('</think>')|last).lstrip('\n') }}], { m => "x</think>\n\ny" }), 'y', 'a filter in parentheses');
my $tr = q[{% if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}no]
       . q[{% else %}yes{% endif %}];
is(r($tr, { content => '<tool_response>x</tool_response>' }), 'yes', 'startswith and endswith: true');
is(r($tr, { content => 'x</tool_response>' }), 'no', 'startswith: false');
is(r(q[{% if not reasoning_content.strip() == '' %}t{% else %}f{% endif %}], { reasoning_content => " \n " }), 'f',
   'strip');
is(r(q[{{ tool.get('name') }}|{{ tool.get('nope', 'dflt') }}], { tool => { name => 'f' } }), 'f|dflt', 'dict get');
is(r(q[{{ 'aXbXc'.replace('X', '-', 1) }} {{ 'Ab'.upper() }}{{ 'Ab'.lower() }}]), 'a-bXc ABab',
   'replace, upper, lower');
is(r(q[text.split('x') {{ "s.strip('.')".strip('"') }}]), q[text.split('x') s.strip('.')],
   'text outside tags and inside string literals stays');
is(r(q[{{ args|tojson }}], { args => { city => 'Gent', n => 2, l => [1, 'été'] } }),
   '{"city": "Gent", "l": [1, "été"], "n": 2}', 'tojson as json.dumps writes it');
is(r(q[{% if enable_thinking is false %}off{% endif %}{% if add_generation_prompt %}+{% endif %}],
     { enable_thinking => 0, add_generation_prompt => 1 }), 'off+', 'boolean template flags');
is(r(q[{{ a }} {{ b }} {{ b.strip() }}], { a => 'été', b => ' 中文 ' }), 'été  中文  中文', 'UTF-8 text in and out');

ok(!eval { r(q[{{ raise_exception('System message must be at the beginning.') }}]) }, 'raise_exception aborts');
like($@, qr/System message must be at the beginning/, "with the template's message");

done_testing;
