use strict;
use warnings;
use ExtUtils::MakeMaker;
use File::Spec;
use Cwd qw(abs_path);

# libinference, the C library this module maps (ROCm, a gfx1151 GPU), first found of:
#   1. ROCM_INFERENCE_PREFIX=<dir>: an installed copy, as `make install PREFIX=<dir>` in rocm-hip writes it
#      (<dir>/include/rocm_inference, <dir>/lib)
#   2. pkg-config rocm-inference: an installed copy on the pkg-config path
#   3. the rocm-hip checkout this module sits in (perl/ROCm-Inference; also from a dist unpacked below it, as `make
#      disttest` does): src/ and build/ there; libinference.so is built with `make` (needs hipcc) if missing
# The module is linked with an rpath to that library directory, so it runs without LD_LIBRARY_PATH.
sub find_lib {
    if (my $prefix = $ENV{ROCM_INFERENCE_PREFIX}) {
        return ("$prefix/include/rocm_inference", "$prefix/lib", "ROCM_INFERENCE_PREFIX=$prefix");
    }
    my $cflags = `pkg-config --cflags-only-I rocm-inference 2>/dev/null`;
    my $libdir = `pkg-config --variable=libdir rocm-inference 2>/dev/null`;
    if ($? == 0 && $libdir) {
        chomp($cflags, $libdir);
        (my $inc = $cflags) =~ s/^\s*-I//;
        $inc =~ s/\s.*//;
        return ($inc, $libdir, 'pkg-config rocm-inference') if -f "$inc/inference.h";
    }
    for my $up (2 .. 4) {
        my $root = abs_path(File::Spec->catdir(('..') x $up)) or next;
        next unless -f "$root/src/inference.h" && -f "$root/src/libinference.map" && -f "$root/Makefile";
        unless (-e "$root/build/libinference.so") {
            print "Building libinference.so in $root (make build/libinference.so.1) ...\n";
            system('make', '-C', $root, '-j', 'build/libinference.so.1') == 0
                or die "building libinference in $root failed (needs ROCm / hipcc)\n";
        }
        return ("$root/src", "$root/build", "the rocm-hip checkout $root");
    }
    return;
}

my ($inc, $lib, $from) = find_lib();
# Without the library there is nothing to build: say what is needed and exit 0 without writing a Makefile, which
# CPAN clients and testers report as "not applicable" here rather than as a failure
my $missing = !$inc                          ? "libinference not found"
            : !-f "$inc/inference.h"         ? "inference.h not found in $inc ($from)"
            : !-e "$lib/libinference.so"     ? "libinference.so not found in $lib ($from)"
            :                                  '';
if ($missing) {
    print STDERR <<"EOT";
$missing.
ROCm::Inference needs libinference, the C library of https://github.com/CowboyTim/rocm-hip (ROCm, an AMD gfx1151
GPU): build and install it there (make && make install PREFIX=<dir>), then set ROCM_INFERENCE_PREFIX=<dir> or put
<dir>/lib/pkgconfig on PKG_CONFIG_PATH, or build this module inside a rocm-hip checkout (perl/ROCm-Inference).
EOT
    exit 0;
}
print "Using libinference from $from: $lib\n";

print <<'EOT';
Tests: t/00-load.t runs anywhere; the others need the GPU and a model, given by environment variables:
  ROCM_INFERENCE_TEST_MODEL=<model.gguf>    any supported model (e.g. Qwen3.5-4B-ROCMFP4.gguf): generation,
                                            chat (with Minijinja), the KV block store
  ROCM_INFERENCE_TEST_QWEN38=<first shard>  Qwen3.8-Flash-Next (qwen4exp), with
  ROCM_INFERENCE_TEST_DRAFT=<mtp head.gguf> its MTP head: speculative decoding
EOT

WriteMakefile(
    NAME               => 'ROCm::Inference',
    VERSION_FROM       => 'lib/ROCm/Inference.pm',
    ABSTRACT_FROM      => 'lib/ROCm/Inference.pm',
    AUTHOR             => 'Tim Aerts <aardbeiplantje@gmail.com>',
    LICENSE            => 'unrestricted',  # the Unlicense (public domain), as the rocm-hip repository
    MIN_PERL_VERSION   => '5.010',
    INC                => "-I$inc",
    LIBS               => ["-L$lib -linference"],
    dynamic_lib        => { OTHERLDFLAGS => "-Wl,-rpath,$lib" },
    CONFIGURE_REQUIRES => { 'ExtUtils::MakeMaker' => '6.64' },
    PREREQ_PM          => { 'Carp' => 0, 'Exporter' => 0, 'XSLoader' => 0 },
    EXE_FILES          => ['bin/rocm-inference-server'],
    TEST_REQUIRES      => { 'Test::More' => '0.88', 'File::Temp' => 0 },
    META_MERGE         => {
        'meta-spec' => { version => 2 },
        prereqs     => {
            runtime => { recommends => { 'Minijinja' => '0.1.1', 'Web::Route' => '0.01' } },
            develop => { requires => { 'Test::Pod' => '1.22', 'Test::Pod::Coverage' => '1.08' } },
        },
        resources => {
            license    => ['https://unlicense.org/'],
            repository => {
                type => 'git',
                url  => 'https://github.com/CowboyTim/rocm-hip.git',
                web  => 'https://github.com/CowboyTim/rocm-hip/tree/main/perl/ROCm-Inference',
            },
            bugtracker => { web => 'https://github.com/CowboyTim/rocm-hip/issues' },
        },
    },
    clean              => { FILES => 'ROCm-Inference-*' },
);
