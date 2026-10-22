use strict;
use warnings;
use ExtUtils::MakeMaker;

WriteMakefile(
    NAME               => 'Web::Route',
    VERSION_FROM       => 'lib/Web/Route.pm',
    ABSTRACT_FROM      => 'lib/Web/Route.pm',
    AUTHOR             => 'Tim Aerts <aardbeiplantje@gmail.com>',
    LICENSE            => 'unrestricted',  # the Unlicense (public domain)
    MIN_PERL_VERSION   => '5.014',
    CONFIGURE_REQUIRES => { 'ExtUtils::MakeMaker' => '6.64' },
    PREREQ_PM          => {
        'Carp' => 0, 'Exporter' => 0, 'IO::Select' => 0, 'IO::Socket::UNIX' => 0, 'IO::Socket::IP' => 0,
        'JSON::PP' => '2.0', 'Socket' => 0, 'Time::HiRes' => 0,
    },
    TEST_REQUIRES      => { 'Test::More' => '0.88', 'File::Temp' => 0, 'HTTP::Tiny' => 0, 'POSIX' => 0 },
    META_MERGE         => {
        'meta-spec' => { version => 2 },
        prereqs     => { develop => { requires => { 'Test::Pod' => '1.22', 'Test::Pod::Coverage' => '1.08' } } },
        resources   => {
            license    => ['https://unlicense.org/'],
            repository => {
                type => 'git',
                url  => 'https://github.com/CowboyTim/rocm-hip.git',
                web  => 'https://github.com/CowboyTim/rocm-hip/tree/main/perl/Web-Route',
            },
            bugtracker => { web => 'https://github.com/CowboyTim/rocm-hip/issues' },
        },
    },
    clean              => { FILES => 'Web-Route-*' },
);
