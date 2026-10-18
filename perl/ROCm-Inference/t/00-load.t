use strict;
use warnings;
use Test::More;

use_ok('ROCm::Inference', qw(:all));

# No model needed: constants and the error path
ok(defined &ROCm::Inference::load_model, 'load_model is defined');
is(ROCm::Inference::STOP_EOG(), 1, 'STOP_EOG');
is(ROCm::Inference::STOP_CONTEXT_FULL(), 5, 'STOP_CONTEXT_FULL');
my $err = eval { ROCm::Inference::load_model('/nonexistent/model.gguf'); 1 } ? '' : $@;
like($err, qr/load_model/, 'loading a missing file dies with a message');
my $bad = eval { ROCm::Inference::n_ctx(\"not a handle"); 1 } ? '' : $@;
like($bad, qr/not a model handle/, 'a non-handle argument dies');

done_testing;
