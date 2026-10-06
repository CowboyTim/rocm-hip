#!/usr/bin/perl

use strict; use warnings;

use Sys::Mmap;
use Fcntl;
use Encode;

my $gguf_fn = shift @ARGV;
$gguf_fn and -e $gguf_fn and -s $gguf_fn or die "usage: $0 <file>\n";

my $gguf = gguf_file($gguf_fn)
    // die "problem validating $gguf_fn\n";
print "gguf $gguf_fn loaded\n";


sub _handle_t {
    my ($fh, $t, $o) = @_;
    no strict "refs";
    my $m = "_t_$t";
    return &{$m}($fh, $o);
}

sub _t_0 {
    my ($fh, $o) = @_;
    return unpack('C', _read_gguf($fh, $o, 1)), 1;
}
sub _t_1 {
    my ($fh, $o) = @_;
    return unpack('c', _read_gguf($fh, $o, 1)), 1;
}
sub _t_2 {
    my ($fh, $o) = @_;
    return unpack('S<', _read_gguf($fh, $o, 2)), 2;
}
sub _t_3 {
    my ($fh, $o) = @_;
    return unpack('s<', _read_gguf($fh, $o, 2)), 2;
}
sub _t_4 {
    my ($fh, $o) = @_;
    return unpack('L<', _read_gguf($fh, $o, 4)), 4;
}
sub _t_5 {
    my ($fh, $o) = @_;
    return unpack('l<', _read_gguf($fh, $o, 4)), 4;
}
sub _t_6 {
    my ($fh, $o) = @_;
    return unpack('f<', _read_gguf($fh, $o, 4)), 4;
}
sub _t_7 {
    my ($fh, $o) = @_;
    return ((0 == unpack("c", _read_gguf($fh, $o, 1))) ? 0 : 1), 1;
}
sub _t_8 {
    my ($fh, $o) = @_;
    my $l = unpack('Q<', _read_gguf($fh, $o, 8));
    my $s = unpack("a*", _read_gguf($fh, undef, $l));
    Encode::_utf8_off($s);
    return $s, 8+$l;
}
sub _t_9 {
    my ($fh, $o) = @_;
    my ($t, $n) = unpack('L<Q<', _read_gguf($fh, $o, 12));
    my $list = [];
    my $c = 0;
    for(my $i = 0; $i < $n; $i++){
        my ($v, $p) = _handle_t($fh, $t);
        $c+=$p;
        push @$list, $v;
    }
    return $list, 12+$c;
}
sub _t_10 {
    my ($fh, $o) = @_;
    return unpack('Q<', _read_gguf($fh, $o, 8)), 8;
}
sub _t_11 {
    my ($fh, $o) = @_;
    return unpack('q<', _read_gguf($fh, $o, 8)), 8;
}
sub _t_12 {
    my ($fh, $o) = @_;
    return unpack('d<', _read_gguf($fh, $o, 8)), 8;
}

sub _read_gguf {
    my ($fh, $o, $l) = @_;
    if(defined $o){
        sysseek($fh, $o, 0)
            or die "Error seek to $o: $!\n";
    }
    my $r = sysread($fh, my $d, $l, 0)
        // die "Error reading $l bytes: $!\n";
    $r == $l
        or die "Error read size ($l): $!\n";
    return $d;
}

sub gguf_file {
    my ($fn) = @_;
    sysopen(my $fh, $fn, O_RDONLY)
        // die "Error opening $fn: $!\n";
    my $gguf = bless {-fh => $fh, -fn => $fn}, "gguf";
    my $gguf_magic = _read_gguf($fh, 0, 4);
    # header
    print "GGUF Magic:   $gguf_magic\n";
    return unless $gguf_magic eq "GGUF";
    my ($gguf_version, $nr_tensors, $nr_kv_metadata) = unpack('L<Q<Q<', _read_gguf($fh, 4, 20));
    print "GGUF Version: $gguf_version\n";
    print "tensor count: $nr_tensors\n";
    print "nr kv meta: $nr_kv_metadata\n";
    sysseek($fh, 24, 0)
        or die "Error seek to 24: $!\n";

    # header kv metadata
    my $p;
    for(my $i = 0; $i < $nr_kv_metadata; $i++){
        (my $k, $p) = _handle_t($fh, 8);
        print "$i.K: $k\n";
        (my $t, $p) = _handle_t($fh, 4);
        print "$i.T: $t\n";
        (my $v, $p) = _handle_t($fh, $t);
        if(ref($v) eq "ARRAY"){
            print "$i.V: $_\n" for join('', @{$v}[0,1,2]);
        } else {
            print "$i.V: $v\n";
        }
    }

    # tensors
    for(my $i = 0; $i < $nr_tensors; $i++){
        (my $v, $p) = _handle_t($fh, 8);
        print "T.$i.NAME: $v\n";
        (my $d, $p) = _handle_t($fh, 4);
        print "T.$i.DIM: $d\n";
        my $dimensions = [];
        for(my $j = 0; $j < $d; $j++){
            (my $dim, $p) = _handle_t($fh, 10);
            push @$dimensions, $dim;
        }
        print "T.$i.DIMS: ".join(',', @$dimensions)."\n";
        (my $ggml_type, $p) = _handle_t($fh, 4);
        print "T.$i.GGML_TYPE: $d\n";
        (my $tensor_offset, $p) = _handle_t($fh, 10);
        print "T.$i.OFFSET: $tensor_offset\n";
    }

    return $gguf;
}
