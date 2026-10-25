package ROCm::Inference;

use strict;
use warnings;

use Carp qw(croak);
use Exporter 'import';

our $VERSION;

BEGIN {  # at compile time: the constants below call into the XS
    $VERSION = '0.01';
    require XSLoader;
    XSLoader::load('ROCm::Inference', $VERSION);
}

# The C library (libinference.so) mapped one function per C call; plus the plain-Perl chat helpers below
our @EXPORT_OK = qw(
    load_model free_model kv_cache_init set_lm_head_q8 set_draft_model set_draft_tokens set_prefill_batch
    tokenize detokenize chat_template bos_id eos_id token_is_eog vocab_size
    n_ctx n_past reset prefill decode_token get_logits
    generate last_prompt_stats last_error
    store_open store_close store_stats store_block_names
    chat_render chat reply_tags
    STOP_NONE STOP_EOG STOP_MAX_TOKENS STOP_STRING STOP_CALLBACK STOP_CONTEXT_FULL
);
our %EXPORT_TAGS = (all => \@EXPORT_OK);

use constant {
    STOP_NONE         => _constant_stop(0),
    STOP_EOG          => _constant_stop(1),
    STOP_MAX_TOKENS   => _constant_stop(2),
    STOP_STRING       => _constant_stop(3),
    STOP_CALLBACK     => _constant_stop(4),
    STOP_CONTEXT_FULL => _constant_stop(5),
};

# Python methods that chat templates call on strings and dicts (content.startswith('<tool_response>'),
# content.split('</think>')[-1].lstrip('\n'), tool.get('name'), ...). Minijinja's C API has no pycompat layer,
# so _py_methods rewrites RECV.name(args) inside template tags to (RECV|py_name(args)), and these filters give
# the Python semantics; a name mapped to a string uses that builtin filter instead.
sub _py_strip_re { defined $_[0] ? '[' . quotemeta($_[0]) . ']' : '\s' }
sub _py_bool { require JSON::PP; $_[0] ? JSON::PP::true() : JSON::PP::false() }
our %PY_METHODS = (
    startswith => sub {
        my ($s, $p) = (shift // '', shift);
        _py_bool(grep { index($s, $_) == 0 } ref $p eq 'ARRAY' ? @$p : ($p // ''));
    },
    endswith => sub {
        my ($s, $p) = (shift // '', shift);
        _py_bool(grep { length($s) >= length && substr($s, length($s) - length) eq $_ } ref $p eq 'ARRAY' ? @$p : ($p // ''));
    },
    split => sub {
        my ($s, $sep, $max) = (shift // '', @_);
        my $limit = defined $max && $max >= 0 ? $max + 1 : -1;
        return [ split /\Q$sep\E/, $s, $limit ] if defined $sep;
        $s =~ s/^\s+//;
        [ split /\s+/, $s, $limit < 0 ? 0 : $limit ];
    },
    strip  => sub { my ($s, $re) = (shift // '', _py_strip_re(shift)); $s =~ s/^$re+//; $s =~ s/$re+\z//; $s },
    lstrip => sub { my ($s, $re) = (shift // '', _py_strip_re(shift)); $s =~ s/^$re+//; $s },
    rstrip => sub { my ($s, $re) = (shift // '', _py_strip_re(shift)); $s =~ s/$re+\z//; $s },
    upper  => sub { uc(shift // '') },
    lower  => sub { lc(shift // '') },
    replace => sub {
        my ($s, $old, $new, $n) = (shift // '', shift // '', shift // '', shift);
        return $s if $old eq '';
        my $out = '';
        while ((!defined $n || $n-- > 0) && (my $i = index($s, $old)) >= 0) {
            $out .= substr($s, 0, $i) . $new;
            $s = substr($s, $i + length $old);
        }
        $out . $s;
    },
    get => sub {
        my ($h, $k, $def) = @_;
        ref $h eq 'HASH' && defined $k && defined $h->{$k} ? $h->{$k} : $def;
    },
    items => 'items',
);

# The tojson filter (tool call arguments, tool definitions) as Python's json.dumps writes it: ", " and ": "
# separators, non-ASCII kept; keys sorted, as Perl hashes have no order
sub _py_tojson {
    my ($v, $indent) = @_;
    my $js = JSON::PP->new->canonical->allow_nonref->allow_blessed;
    return $js->indent->indent_length($indent)->space_after->encode($v) =~ s/\n\z//r if $indent;
    $js->space_after->encode($v) =~ s/("(?:[^"\\]|\\.)*")|,/$1 \/\/ ', '/ger;
}

# The method calls in a template's {{ }} and {% %} tags as filter calls (see %PY_METHODS)
sub _py_methods {
    my ($tmpl) = @_;
    my $names = join '|', keys %PY_METHODS;
    my $tag = sub {
        my ($s) = @_;
        while (1) {
            my @t = $s =~ /\G('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|\w+|\s+|.)/gs;
            my ($j) = grep { $t[$_] eq '.' && ($t[$_ + 1] // '') =~ /^(?:$names)\z/ && ($t[$_ + 2] // '') eq '(' } 0 .. $#t;
            return $s unless defined $j;
            # the receiver: names, attributes, subscripts, calls and string literals left of the dot
            my $k = $j;
            while ($k > 0) {
                my $p = $t[$k - 1];
                if ($p eq ')' || $p eq ']') {
                    my ($close, $open, $d) = ($p, $p eq ')' ? '(' : '[', 0);
                    do { --$k; $d += ($t[$k] eq $close) - ($t[$k] eq $open) } while ($d && $k > 0);
                } elsif ($p =~ /^(?:\w+|\.|['"].*)\z/s) {
                    --$k;
                } else {
                    last;
                }
            }
            my ($e, $d) = ($j + 2, 0);  # the closing parenthesis of the call
            for (; $e <= $#t; ++$e) { $d += ($t[$e] eq '(') - ($t[$e] eq ')'); last unless $d }
            my $f = $PY_METHODS{ $t[$j + 1] };
            $s = join '', @t[0 .. $k - 1], '(', @t[$k .. $j - 1], '|', ref $f ? "py_$t[$j + 1]" : $f,
                @t[$j + 2 .. $e], ')', @t[$e + 1 .. $#t];
        }
    };
    $tmpl =~ s/(\{\{|\{%)(.*?)(\}\}|%\})/$1 . $tag->($2) . $3/gse;
    return $tmpl;
}

# Minijinja 0.1.1: callbacks that die abort the rendering, blessed (boolean) returns, Perl character strings
our $MINIJINJA_VERSION = '0.1.1';
sub _require_minijinja { eval { require Minijinja; Minijinja->VERSION($MINIJINJA_VERSION); 1 } }

# chat_render($model, \@messages, %vars): the conversation as prompt text, rendered with the model's own chat
# template (the GGUF's tokenizer.chat_template) by Minijinja. Template variables: messages, bos_token,
# eos_token, add_generation_prompt (default 1), and any %vars (e.g. enable_thinking => 0, tools => [...]);
# template_name => 'name' picks tokenizer.chat_template.<name>. When tokenize adds a BOS (Gemma), the text
# goes without the template's leading one, so that tokenize($m, $text) has it once.
sub chat_render {
    my ($m, $messages, %vars) = @_;
    _require_minijinja() or croak "ROCm::Inference::chat_render needs the Minijinja module $MINIJINJA_VERSION: $@";
    my $name = delete $vars{template_name};
    my $tmpl = defined $name ? chat_template($m, $name) : chat_template($m);
    croak "ROCm::Inference::chat_render: the model has no chat template" . ($name ? " '$name'" : '')
        unless defined $tmpl;
    my ($bos, $eos) = (bos_id($m), eos_id($m));
    my %ctx = (
        messages              => $messages,
        add_generation_prompt => 1,
        bos_token             => $bos >= 0 ? detokenize($m, [$bos]) : '',
        eos_token             => $eos >= 0 ? detokenize($m, [$eos]) : '',
        %vars,
    );
    my $text = _render_template($tmpl, \%ctx);
    # tokenize adds the BOS itself when the model wants one (Gemma): drop the template's, as llama.cpp does
    my $add = tokenize($m, '');
    substr($text, 0, length $ctx{bos_token}, '')
        if length $ctx{bos_token} && @$add && $add->[0] == $bos && index($text, $ctx{bos_token}) == 0;
    return $text;
}

# A chat template rendered by Minijinja with the Python methods (%PY_METHODS) and tojson
sub _render_template {
    my ($tmpl, $vars) = @_;
    my %ctx = %$vars;
    # Perl has no booleans, and templates test these with 'is false' / 'is true'
    require JSON::PP;
    for (grep { exists $ctx{$_} && !ref $ctx{$_} } qw(add_generation_prompt enable_thinking)) {
        $ctx{$_} = $ctx{$_} ? JSON::PP::true() : JSON::PP::false();
    }
    my $env = Minijinja::minijinja();
    Minijinja::add_filter($env, "py_$_", $PY_METHODS{$_}) for grep { ref $PY_METHODS{$_} } keys %PY_METHODS;
    Minijinja::add_filter($env, 'tojson', \&_py_tojson);
    Minijinja::add_function($env, 'raise_exception', sub { die(($_[0] // 'template error') . "\n") });
    my $text = Minijinja::render_str($env, 'chat_template', _py_methods($tmpl), \%ctx);
    croak "ROCm::Inference::chat_render: " . (Minijinja::error_detail() // 'template error') unless defined $text;
    return $text;
}

# The tags that mark a reply's reasoning and its tool calls; everything else is content. reasoning's third
# entry is the text that closes it when max_thinking_tokens runs out. %REPLY_TAGS is the Qwen / Nemotron set
# (the default); %REPLY_TAG_SETS has one set per chat format, and reply_tags($m) picks the model's
our %REPLY_TAGS = (reasoning => ['<think>', '</think>', "\n</think>\n\n"], tool_call => ['<tool_call>', '</tool_call>']);
our %REPLY_TAG_SETS = (
    qwen   => \%REPLY_TAGS,
    # Gemma-4: <|channel>thought\n...<channel|>, <|tool_call>call:name{key:value,...}<tool_call|>
    gemma4 => { reasoning => ['<|channel>thought', '<channel|>', "\n<channel|>"],
                tool_call => ['<|tool_call>', '<tool_call|>'] },
);

# reply_tags($m): the tag set (%REPLY_TAG_SETS) of the model's chat format, from the tags its chat template
# writes; the default %REPLY_TAGS when there is no template or no other set matches
sub reply_tags {
    my ($m) = @_;
    my $tmpl = eval { chat_template($m) } // '';
    for my $name (grep { $_ ne 'qwen' } sort keys %REPLY_TAG_SETS) {
        my $set = $REPLY_TAG_SETS{$name};
        return $set unless grep { index($tmpl, $_) < 0 } map { @{$_}[0, 1] } values %$set;
    }
    return \%REPLY_TAGS;
}

# A splitter for a reply's text as it streams: feed($text) takes the next piece (true when on_delta asked to
# stop), finish() the end. Pieces go to on_delta($text, $kind) by kind: 'reasoning', 'content' or 'tool_call'
# (a tool call block's inner text), without the tags. Reasoning and content are trimmed as the chat templates
# trim them (leading whitespace dropped, trailing whitespace held back until more text follows), so the
# deltas of a kind add up to the message's field. Tags are found in the text, not by token: the text a token
# brings may arrive with the next one.
sub _reply_splitter {
    my ($kind, $on_delta, $tags) = @_;
    $tags //= \%REPLY_TAGS;
    my (%text, %hold, @calls, $buf, $stop) = (reasoning => '', content => '');
    $buf = '';
    my $put = sub {
        my ($k, $t) = @_;
        if ($k eq 'tool_call') {
            $calls[-1] .= $t;
        } else {
            $t = $hold{$k} . $t if defined $hold{$k};
            $t =~ s/^\s+// unless length $text{$k};
            $t =~ s/(\s*)\z//;
            $hold{$k} = $1;
            $text{$k} .= $t;
        }
        $stop = 1 if length $t && $on_delta && $on_delta->($t, $k);
    };
    my $feed = sub {
        $buf .= $_[0] // '';
        while (length $buf) {
            # the tags that can end this kind: its close tag, or in content the open tags
            my %next = $kind eq 'content' ? map { $tags->{$_}[0] => $_ } keys %$tags : ($tags->{$kind}[1] => 'content');
            my ($at, $tag) = (-1, undef);
            for my $t (keys %next) {
                my $i = index($buf, $t);
                ($at, $tag) = ($i, $t) if $i >= 0 && ($at < 0 || $i < $at);
            }
            if ($at < 0) {  # keep what could be the start of a tag
                my $keep = 0;
                for my $t (keys %next) {
                    my $n = length($t) - 1;
                    $n = length $buf if $n > length $buf;
                    for (; $n > $keep; --$n) {
                        if (substr($buf, -$n) eq substr($t, 0, $n)) { $keep = $n; last }
                    }
                }
                $put->($kind, substr($buf, 0, length($buf) - $keep, '')) if length($buf) > $keep;
                last;
            }
            $put->($kind, substr($buf, 0, $at)) if $at > 0;
            substr($buf, 0, $at + length $tag, '');
            $kind = $next{$tag};
            push @calls, '' if $kind eq 'tool_call';
        }
        my $s = $stop;
        $stop = 0;
        return $s;
    };
    return {
        feed   => $feed,
        kind   => sub { $kind },
        finish => sub {
            $put->($kind, $buf) if length $buf;
            $buf = '';
            my %msg = (role => 'assistant', content => $text{content});
            $msg{reasoning_content} = $text{reasoning} if length $text{reasoning};
            for my $raw (@calls) {
                my $call = _parse_tool_call($raw);
                push @{ $msg{$call ? 'tool_calls' : 'invalid_tool_calls'} }, $call // $raw;
            }
            return \%msg;
        },
    };
}

# A tool call block's text as { type => 'function', function => { name, arguments => {...} } }: Qwen3.5+ XML
# (<function=name> <parameter=p>value</parameter>, values as written), Qwen3 JSON ({"name", "arguments"}) or
# Gemma-4 (call:name{key:value,...}); undef when it is none of those
sub _parse_tool_call {
    my ($t) = @_;
    if ($t =~ /^\s*call:([^{\s]+)\{/gc) {
        my $name = $1;
        pos($t) -= 1;
        my ($args) = _parse_gemma_value(\$t);
        return undef unless ref $args eq 'HASH' && $t =~ /\G\s*\z/gc;
        return { type => 'function', function => { name => $name, arguments => $args } };
    }
    if ($t =~ /<function=([^>\n]+)>(.*?)(?:<\/function>|\z)/s) {
        my ($name, $body, %args) = ($1, $2);
        $args{$1} = $2 while $body =~ /<parameter=([^>\n]+)>\n?(.*?)\n?<\/parameter>/sg;
        return { type => 'function', function => { name => $name, arguments => \%args } };
    }
    require JSON::PP;
    my $j = eval { JSON::PP->new->decode($t) };
    return undef unless ref $j eq 'HASH' && defined $j->{name} && !ref $j->{name};
    my $args = $j->{arguments} // {};
    $args = eval { JSON::PP->new->decode($args) } // $args unless ref $args;  # arguments as a JSON string
    return { type => 'function', function => { name => $j->{name}, arguments => $args } };
}

# Gemma-4's argument syntax, from pos($$t) on: <|"|>string<|"|>, {key:value,...} (keys bare or quoted),
# [value,...], true / false / null, numbers, and other bare text up to the next , } or ] as a string. Returns
# the value as a one-element list; an empty list (and pos unchanged) when it does not parse
my $GQ = '<|"|>';
sub _parse_gemma_value {
    my ($t) = @_;
    my $at = pos($$t) // 0;
    my $end = qr/(?=\s*(?:[,}\]]|\z))/;
    $$t =~ /\G\s*/gc;
    return $1 if $$t =~ /\G\Q$GQ\E(.*?)\Q$GQ\E/gcs;
    if ($$t =~ /\G([{\[])/gc) {
        my ($obj, %h, @a) = ($1 eq '{');
        my $close = $obj ? '}' : ']';
        return $obj ? \%h : \@a if $$t =~ /\G\s*\Q$close\E/gc;
        while (1) {
            my $k;
            if ($obj) {
                $$t =~ /\G\s*/gc;
                if ($$t =~ /\G\Q$GQ\E(.*?)\Q$GQ\E\s*:/gcs || $$t =~ /\G([^:{}\[\],]+?)\s*:/gc) { $k = $1 }
                else { last }
            }
            my @v = _parse_gemma_value($t) or last;
            $obj ? ($h{$k} = $v[0]) : push @a, $v[0];
            next if $$t =~ /\G\s*,/gc;
            return $obj ? \%h : \@a if $$t =~ /\G\s*\Q$close\E/gc;
            last;
        }
    } elsif ($$t =~ /\G(true|false|null)$end/gc) {
        require JSON::PP;
        return $1 eq 'null' ? undef : $1 eq 'true' ? JSON::PP::true() : JSON::PP::false();
    } elsif ($$t =~ /\G(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?)$end/gc) {
        return 0 + $1;
    } elsif ($$t =~ /\G([^,{}\[\]]+?)(?=\s*[,}\]])/gc) {
        return $1;
    }
    pos($$t) = $at;
    return;
}

# chat($model, \@messages, \%options): render the conversation, then generate the assistant's reply, reusing
# what the model already holds of the same conversation (reuse_prefix). The reply is split into its reasoning
# (the <think> block), its content and its tool calls. Returns the content, and in list context also the stop
# reason and the reply as a message ({ role => 'assistant', content, reasoning_content, tool_calls }) to push
# onto @messages for the next turn. %options: those of generate (on_token, max_new_tokens, temperature, stop,
# ...), template variables in vars => { enable_thinking => 0, tools => [...], ... }, and
#   on_delta => sub { my ($text, $kind) = @_; return $stop }   the reply streamed by kind (see _reply_splitter)
#   max_thinking_tokens => $n   close the reasoning after $n tokens and go on with the answer
sub chat {
    my ($m, $messages, $opts) = @_;
    my %o = %{ $opts // {} };
    my %vars = %{ delete $o{vars} // {} };
    my ($on_delta, $budget, $on_token) = delete @o{qw(on_delta max_thinking_tokens on_token)};
    my $text = chat_render($m, $messages, %vars);
    # Does the generation prompt open the reasoning (Qwen3.5+ with thinking on ends it with "<think>\n")?
    my $tags = reply_tags($m);
    my ($open, $close, $closing) = @{ $tags->{reasoning} };
    my $history = eval { chat_render($m, $messages, %vars, add_generation_prompt => 0) } // '';
    my $gen = index($text, $history) == 0 ? substr($text, length $history) : $text;
    my $split = _reply_splitter(rindex($gen, $open) > rindex($gen, $close) ? 'reasoning' : 'content', $on_delta,
                                $tags);

    my ($max, $thinking, @ids, $reason) = ($o{max_new_tokens} // 256, 0);
    my %g = (reuse_prefix => 1, %o);
    my $prompt = tokenize($m, $text);
    while (1) {
        my ($stopped, $budget_hit) = (0, 0);
        $g{max_new_tokens} = $max - @ids;
        $g{on_token} = sub {
            my ($tok, $piece) = @_;
            $stopped = 1 if $on_token && $on_token->($tok, $piece);
            $stopped = 1 if $split->{feed}->($piece);
            $budget_hit = 1 if defined $budget && $tok >= 0 && $split->{kind}->() eq 'reasoning' && ++$thinking >= $budget;
            return $stopped || $budget_hit;
        };
        my ($ids, $why) = generate($m, $prompt, \%g);
        push @ids, @$ids;
        $reason = $why;
        last unless $budget_hit && !$stopped && $why == STOP_CALLBACK() && @ids < $max;
        # Out of thinking budget: close the reasoning as the model would and continue with the answer, right
        # after the tokens generated so far
        $closing //= $close;
        $prompt = tokenize($m, $closing);
        $split->{feed}->($closing);
        $budget = undef;
        $g{reuse_prefix} = 0;
    }
    my $msg = $split->{finish}->();
    return wantarray ? ($msg->{content}, $reason, $msg) : $msg->{content};
}

1;

__END__

=head1 NAME

ROCm::Inference - LLM inference on AMD Strix Halo (gfx1151) through libinference

=head1 SYNOPSIS

    use ROCm::Inference qw(:all);

    my $m = load_model('/dev/shm/models/Qwen3.5-4B-ROCMFP4.gguf');
    kv_cache_init($m, 8192);

    # Plain generation, streamed
    my ($ids, $why) = generate($m, tokenize($m, 'The capital of France is'), {
        max_new_tokens => 32,
        temperature    => 0,
        on_token       => sub { my ($token, $text) = @_; print $text; return 0 },
    });

    # Chat: the model's own template, rendered with Minijinja; each turn reuses the previous ones
    my @messages = ({ role => 'user', content => 'Name three primary colors.' });
    my $reply = chat($m, \@messages, { vars => { enable_thinking => 0 } });
    push @messages, { role => 'assistant', content => $reply },
                    { role => 'user', content => 'And three secondary colors?' };
    print chat($m, \@messages, { vars => { enable_thinking => 0 } }), "\n";

    # Thinking, streamed by kind ('reasoning', 'content', 'tool_call'), at most 256 tokens of it; the reply
    # message keeps the reasoning and any tool calls for the next turn
    my ($answer, $why, $msg) = chat($m, \@messages, {
        vars                => { enable_thinking => 1, tools => \@tools },
        max_thinking_tokens => 256,
        on_delta            => sub { my ($text, $kind) = @_; print $kind eq 'content' ? $text : ''; return 0 },
    });
    push @messages, $msg;    # { role => 'assistant', content, reasoning_content, tool_calls }

=head1 DESCRIPTION

A thin XS mapping of the inference C library (F<src/inference.h> in the rocm-hip repository): one function
per C function, the model handle first. Errors die with the library's message. Text goes in and out as Perl
character strings, token lists as array references. Chat templates are rendered in Perl with L<Minijinja>.

=head1 FUNCTIONS

=over

=item load_model($path) / free_model($m)

Load a GGUF model (memory-mapped, zero-copy). The handle frees itself when it goes out of scope.

=item kv_cache_init($m, $n_ctx)

Allocate the context (KV cache and recurrent state) for C<$n_ctx> tokens. Required before generating.

=item tokenize($m, $text) / detokenize($m, \@ids)

Special tokens written in the text (e.g. C<< <|im_start|> >>) become their ids.

=item generate($m, \@prompt, \%options)

Continues the current sequence with the prompt tokens and generates. Returns C<(\@ids, $stop_reason)>.
Options: C<max_new_tokens> (256), C<temperature> (0.8; 0 = greedy), C<top_k>, C<top_p>, C<min_p>,
C<repeat_penalty>, C<repeat_last_n>, C<seed>, C<stop> (array ref of strings), C<ignore_eog>,
C<reuse_prefix> (the prompt is the whole sequence; the shared prefix is reused), C<checkpoint_tail>, and
C<on_token> (called with the token id and the text that is ready; return true to stop).

=item store_open($m, $dir [, $snapshot_every = 4]) / store_close($m)

Attach a KV block store: a directory (e.g. under F</dev/shm>) that holds the KV cache of every full
block of every sequence as a file named by the chained hash of its tokens, plus recurrent-state snapshots
every C<$snapshot_every> blocks and at the end of each prompt. Prompts generated with C<reuse_prefix> then
start from the longest prefix any process stored, mapped in place without copying. The files are plain:
delete them to evict (the library touches them when used), move them to other storage and back.

=item store_stats($m)

A hash ref: C<blocks_published>, C<blocks_attached>, C<snapshots_saved>, C<snapshots_restored>,
C<tokens_from_store>, C<block_tokens>, C<block_bytes>, C<snapshot_bytes>.

=item store_block_names($m, \@ids)

The store file names of the token list's full blocks, in order (e.g. to keep a system prompt from eviction).

=item set_draft_model($m, $path) / set_draft_model($m, undef)

Load an MTP draft head for speculative decoding (Qwen3.8-Flash-Next: a nextn sidecar GGUF such as
F<mtp-Qwen3.8-Flash-Next-Q4_0_ROCMFP4_FAST.gguf>), or unload it. C<generate> then decodes each token
together with the head's guesses of the next ones and keeps each guess that is the token sampled anyway: the
same tokens as without the head, for any sampling options, up to three times as many per pass when the head
guesses well. Dies when the file is not a head for this model. Before or after C<kv_cache_init>.

=item set_prefill_batch($m, $n_tokens)

Tokens per prefill micro-batch, 16 to 8192, or 0 for the default (1024). Larger batches speed up long prompts
on MoE models (up to ~10%: each expert's weights are read once per batch) at the cost of scratch memory (about
1 MB per token on Qwen3.8-Flash-Next); no gain for dense models. Before C<kv_cache_init>; dies after it.

=item set_draft_tokens($m, $n)

Guesses per pass with a draft head: 1 to 3, or 0 (the default) to choose 2 or 3 per pass from how well the
head has been guessing (3 pays off on code and chat replies). Dies outside that range.

=item chat_render($m, \@messages, %vars) / chat($m, \@messages, \%options)

Render a conversation with the model's chat template / render it and generate the reply.
The template is rendered by L<Minijinja> with what chat templates expect of Python's Jinja: string and dict
methods (C<startswith>, C<endswith>, C<split>, C<strip>, C<lstrip>, C<rstrip>, C<replace>, C<upper>,
C<lower>, C<get>, C<items>), C<tojson> as C<json.dumps> writes it, C<raise_exception>, and the plain Perl
values of C<enable_thinking> / C<add_generation_prompt> as booleans. Text is passed and returned as Perl
character strings. Needs Minijinja 0.1.1 or later.

C<chat> splits the reply at its tags into reasoning, content and tool calls: Qwen / Nemotron's
C<< <think> >> ... C<< </think> >> and C<< <tool_call> >> ... C<< </tool_call> >>
(C<%ROCm::Inference::REPLY_TAGS>), or Gemma-4's C<< <|channel>thought >> ... C<< <channel|> >> and
C<< <|tool_call> >> ... C<< <tool_call|> >>; C<reply_tags($m)> picks the set (C<%ROCm::Inference::REPLY_TAG_SETS>)
whose tags the model's chat template writes. A generation prompt that opens the reasoning (Qwen3.5+ with
thinking on) starts the reply in it. It returns the content,
and in list context also the stop reason and the reply as a message to push onto the conversation:
C<< { role => 'assistant', content, reasoning_content, tool_calls } >>, the tool calls as
C<< { type => 'function', function => { name, arguments => { ... } } } >> (Qwen3.5+ XML parameters as the model
wrote them, Qwen3 JSON decoded, Gemma-4's C<< call:name{key:value,...} >> decoded; blocks that parse as none of
those in C<invalid_tool_calls>). The templates render
such a message back as the model wrote it, so the next turn reuses it. Options beyond C<generate>'s:
C<vars> (template variables), C<on_delta> (C<< sub { my ($text, $kind) = @_; return $stop } >>: the reply
streamed by kind, C<reasoning>, C<content> or C<tool_call>, without the tags; reasoning and content trimmed as
the templates trim them, so the deltas of a kind add up to the message's field) and C<max_thinking_tokens>
(after that many reasoning tokens, close the reasoning as the model would and go on with the answer).

=item chat_template($m [, $name]), bos_id, eos_id, token_is_eog, vocab_size, n_ctx, n_past, reset,
prefill, decode_token, get_logits, last_prompt_stats, set_lm_head_q8, set_draft_model, set_draft_tokens,
set_prefill_batch, last_error

As the C functions of the same name (C<inference_*>). C<get_logits> returns packed floats
(C<unpack 'f*'>).

=back

=head1 CONSTANTS

C<STOP_NONE STOP_EOG STOP_MAX_TOKENS STOP_STRING STOP_CALLBACK STOP_CONTEXT_FULL>: the stop reasons.

=head1 SEE ALSO

L<ROCm::Inference::Store>, L<Minijinja>, L<rocm-inference-server> (an OpenAI-compatible server, with
L<Web::Route>); the C API (F<inference.h>) and the engine:
L<https://github.com/CowboyTim/rocm-hip>.

=head1 AUTHOR

Tim Aerts E<lt>aardbeiplantje@gmail.comE<gt>

=head1 LICENSE

This is free and unencumbered software released into the public domain (the Unlicense): see the LICENSE file
or L<https://unlicense.org/>.

=cut
