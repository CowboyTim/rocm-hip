use strict;
use warnings;
use ExtUtils::MakeMaker;
use File::Spec;
use Cwd qw(abs_path);

# libinference (the C library this module maps): by default the rocm-hip checkout this module lives in
# (../../src for the headers, ../../build for libinference.so after `make` there); or an installed copy
# under ROCM_INFERENCE_PREFIX (include/rocm_inference, lib).
my ($inc, $lib);
if (my $prefix = $ENV{ROCM_INFERENCE_PREFIX}) {
    ($inc, $lib) = ("$prefix/include/rocm_inference", "$prefix/lib");
} else {
    my $root = abs_path(File::Spec->catdir('..', '..'));
    ($inc, $lib) = ("$root/src", "$root/build");
}
-f "$inc/inference.h" or die "inference.h not found in $inc (set ROCM_INFERENCE_PREFIX)\n";
-e "$lib/libinference.so" or die "libinference.so not found in $lib (run `make` in the rocm-hip checkout)\n";

WriteMakefile(
    NAME             => 'ROCm::Inference',
    VERSION_FROM     => 'lib/ROCm/Inference.pm',
    ABSTRACT_FROM    => 'lib/ROCm/Inference.pm',
    AUTHOR           => 'rocm-hip contributors',
    LICENSE          => 'mit',
    MIN_PERL_VERSION => '5.010',
    INC              => "-I$inc",
    LIBS             => ["-L$lib -linference"],
    dynamic_lib      => { OTHERLDFLAGS => "-Wl,-rpath,$lib" },
    PREREQ_PM        => {},
    TEST_REQUIRES    => { 'Test::More' => 0 },
    META_MERGE       => { prereqs => { runtime => { recommends => { 'Minijinja' => 0 } } } },
);
