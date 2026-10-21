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
    chat_render chat
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

# chat_render($model, \@messages, %vars): the conversation as prompt text, rendered with the model's own chat
# template (the GGUF's tokenizer.chat_template) by Minijinja. Template variables: messages, bos_token,
# eos_token, add_generation_prompt (default 1), and any %vars (e.g. enable_thinking => 0, tools => [...]);
# template_name => 'name' picks tokenizer.chat_template.<name>.
sub chat_render {
    my ($m, $messages, %vars) = @_;
    eval { require Minijinja; 1 } or croak "ROCm::Inference::chat_render needs the Minijinja module: $@";
    my $name = delete $vars{template_name};
    my $tmpl = chat_template($m, $name);
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
    my $env = Minijinja::minijinja();
    my $text = Minijinja::render_str($env, 'chat_template', $tmpl, \%ctx);
    croak "ROCm::Inference::chat_render: " . (Minijinja::error_detail() // 'template error') unless defined $text;
    return $text;
}

# chat($model, \@messages, \%options): render the conversation, then generate the assistant's reply, reusing
# what the model already holds of the same conversation (reuse_prefix). Returns the reply text, and in list
# context also the stop reason. %options: those of generate (on_token, max_new_tokens, temperature, stop, ...)
# plus template variables in vars => { enable_thinking => 0, ... }.
sub chat {
    my ($m, $messages, $opts) = @_;
    my %o = %{ $opts // {} };
    my %vars = %{ delete $o{vars} // {} };
    my $prompt = tokenize($m, chat_render($m, $messages, %vars));
    my ($ids, $reason) = generate($m, $prompt, { reuse_prefix => 1, %o });
    my $reply = detokenize($m, $ids);
    return wantarray ? ($reply, $reason) : $reply;
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

=item chat_template($m [, $name]), bos_id, eos_id, token_is_eog, vocab_size, n_ctx, n_past, reset,
prefill, decode_token, get_logits, last_prompt_stats, set_lm_head_q8, set_draft_model, set_draft_tokens,
set_prefill_batch, last_error

As the C functions of the same name (C<inference_*>). C<get_logits> returns packed floats
(C<unpack 'f*'>).

=back

=head1 CONSTANTS

C<STOP_NONE STOP_EOG STOP_MAX_TOKENS STOP_STRING STOP_CALLBACK STOP_CONTEXT_FULL>: the stop reasons.

=cut
