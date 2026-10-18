use strict;
use warnings;
use Test::More;

use_ok('ROCm::Inference', qw(:all));

# No GPU or model needed: the exports, the constants and the error paths
for my $f (grep { !/^STOP_/ } @ROCm::Inference::EXPORT_OK) {
    ok(defined &{"ROCm::Inference::$f"}, "$f is defined");
}
is(ROCm::Inference::STOP_NONE(), 0, 'STOP_NONE');
is(ROCm::Inference::STOP_EOG(), 1, 'STOP_EOG');
is(ROCm::Inference::STOP_CONTEXT_FULL(), 5, 'STOP_CONTEXT_FULL');
my $err = eval { ROCm::Inference::load_model('/nonexistent/model.gguf'); 1 } ? '' : $@;
like($err, qr/load_model/, 'loading a missing file dies with a message');
my %args = (n_ctx => [], set_draft_model => [undef], generate => [[1, 2]]);
for my $f (sort keys %args) {
    my $bad = eval { no strict 'refs'; &{"ROCm::Inference::$f"}(\"not a handle", @{ $args{$f} }); 1 } ? '' : $@;
    like($bad, qr/not a model handle/, "$f with a non-handle argument dies");
}

done_testing;
