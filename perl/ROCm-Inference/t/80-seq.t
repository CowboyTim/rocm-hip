use strict;
use warnings;
use utf8;
use Test::More;

use ROCm::Inference qw(:all);

# Sequences and step-wise generation: two conversations on one model, stepped in turn or batched (one forward pass
# for both), give what each gives alone.
# Needs a GPU, a model and the Minijinja module: ROCM_INFERENCE_TEST_MODEL=<gguf>
my $path = $ENV{ROCM_INFERENCE_TEST_MODEL};
plan skip_all => 'set ROCM_INFERENCE_TEST_MODEL to a GGUF model to run' unless $path && -f $path;
plan skip_all => "Minijinja $ROCm::Inference::MINIJINJA_VERSION is not installed" unless ROCm::Inference::_require_minijinja();
binmode Test::More->builder->$_, ':encoding(UTF-8)' for qw(output failure_output todo_output);

my $m = load_model($path);
kv_cache_init($m, 2048);

# a stream steps through what generate gives
my $p = tokenize($m, 'The capital of France is');
my %g = (max_new_tokens => 12, temperature => 0, ignore_eog => 1);
reset($m);
my ($ids) = generate($m, $p, \%g);
reset($m);
my $st = stream_begin($m, $p, \%g);
my (@got, $text, $going) = ();
do {
    my ($tok, $t);
    ($tok, $t, $going) = $st->step;
    push @got, $tok if $tok >= 0;
    $text .= $t;
} while ($going);
is("@got", "@$ids", 'stream: the tokens of generate');
is($text, detokenize($m, $ids), 'stream: their text');
is($st->tokens, 12, 'stream: tokens');
$st->free;

# two chats alone (sequence 0, from empty), then interleaved on two new sequences
my %o = (temperature => 0, max_new_tokens => 40, vars => { enable_thinking => 0 });
my @q = ([{ role => 'user', content => 'Name three primary colors.' }],
         [{ role => 'user', content => 'What is 7 times 8? Answer with the number.' }]);
my (@alone, @why);
for my $msgs (@q) {
    reset($m);
    my ($content, $why) = chat($m, $msgs, \%o);
    push @alone, $content;
    push @why, $why;
}
my @seqs = map { seq_new($m) } 1 .. 2;
ok($seqs[0] > 0 && $seqs[1] > 0 && $seqs[0] != $seqs[1], "two sequences: @seqs");
my @c;
for my $i (0, 1) {
    seq_select($m, $seqs[$i]);
    push @c, chat_start($m, $q[$i], \%o);
}
is(seq_current($m), $seqs[1], 'seq_current');
my @deltas = ('', '');
my @live = (1, 1);
while ($live[0] || $live[1]) {
    for my $i (0, 1) {
        next unless $live[$i];
        my ($d, $go) = $c[$i]->step;
        $deltas[$i] .= join '', map { $_->[1] eq 'content' ? $_->[0] : '' } @$d;
        $live[$i] = $go;
    }
}
for my $i (0, 1) {
    my ($content, $why, $msg) = $c[$i]->result;
    is($content, $alone[$i], "interleaved chat $i = alone: $content");
    is($deltas[$i], $content, "chat $i: the content deltas add up");
    is($why, $why[$i], "chat $i stops as alone ($why)");
}

# the conversations go on, each with its own history (reused)
seq_select($m, $seqs[0]);
my @next = (@{ $q[0] }, { role => 'assistant', content => $alone[0] }, { role => 'user', content => 'And secondary?' });
my $r = chat($m, \@next, { %o, max_new_tokens => 160 });
my ($reused) = last_prompt_stats($m);
# (recurrent models resume at a saved checkpoint: the prompt's, when the re-rendered reply differs from the tokens)
ok($reused > 0, "sequence $seqs[0] reuses its history: $reused tokens");
like($r, qr/green|orange|purple/i, "and answers: $r");

# the same two chats batched (step_chats: one forward pass for both)
my @b;
for my $i (0, 1) {
    seq_select($m, $seqs[$i]);
    reset($m);
    push @b, chat_start($m, $q[$i], \%o);
}
my @bd = ('', '');
for (my $going = 1; $going;) {
    my @r = step_chats(@b);
    $going = 0;
    for my $i (0, 1) {
        $bd[$i] .= join '', map { $_->[1] eq 'content' ? $_->[0] : '' } @{ $r[$i][0] };
        $going ||= $r[$i][1];
    }
}
for my $i (0, 1) {
    my ($content, $why) = $b[$i]->result;
    is($content, $alone[$i], "batched chat $i = alone");
    is($bd[$i], $content, "batched chat $i: the deltas add up");
}
# and two streams
my @p2 = map { tokenize($m, $_) } 'The capital of France is', 'Water boils at';
my @alone2 = map { reset($m); (generate($m, $_, \%g))[0] } @p2;
my @st2;
for my $i (0, 1) {
    seq_select($m, $seqs[$i]);
    reset($m);
    push @st2, stream_begin($m, $p2[$i], \%g);
}
my @t2 = ([], []);
for (my $going = 1; $going;) {
    my @r = step_streams(@st2);
    $going = 0;
    for my $i (0, 1) {
        push @{ $t2[$i] }, $r[$i][0] if $r[$i][0] >= 0;
        $going ||= $r[$i][2];
    }
}
is("@{ $t2[$_] }", "@{ $alone2[$_] }", "step_streams: stream $_ = generate") for 0, 1;
$_->free for @st2;

seq_free($m, $_) for @seqs;
ok(!eval { seq_select($m, $seqs[0]); 1 }, 'a freed sequence cannot be selected');
ok(!eval { seq_free($m, 0); 1 }, 'sequence 0 cannot be freed');

done_testing;
