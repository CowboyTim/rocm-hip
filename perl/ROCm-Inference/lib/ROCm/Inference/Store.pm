package ROCm::Inference::Store;

use strict;
use warnings;

use Carp qw(croak);
use Exporter 'import';
use File::Copy ();
use File::Spec;

our $VERSION = '0.01';
our @EXPORT_OK = qw(store_usage store_files store_evict store_spill store_fetch);
our %EXPORT_TAGS = (all => \@EXPORT_OK);

# Policy for the KV block store of ROCm::Inference::store_open: plain functions over its directory. The library
# writes <hash>.kvb (KV blocks) and <hash>.rst (state snapshots) and touches them when they are used, so mtime is
# the last use. Deleting or moving files is always safe: a missing file only means that part is prefilled again.

# The store's files: list of { name, path, bytes, mtime }
sub store_files {
    my ($dir) = @_;
    opendir(my $dh, $dir) or croak "ROCm::Inference::Store: cannot open $dir: $!";
    my @f;
    for my $name (readdir $dh) {
        next unless $name =~ /\A[0-9a-f]{64}\.(?:kvb|rst)\z/;
        my $path = File::Spec->catfile($dir, $name);
        my @st = stat $path or next;
        push @f, { name => $name, path => $path, bytes => $st[7], mtime => $st[9] };
    }
    closedir $dh;
    return @f;
}

# (bytes, files) in scalar context bytes
sub store_usage {
    my ($dir) = @_;
    my ($bytes, @f) = (0, store_files($dir));
    $bytes += $_->{bytes} for @f;
    return wantarray ? ($bytes, scalar @f) : $bytes;
}

# keep: array ref of file names (a block's .kvb name protects its .rst too, as store_block_names returns them)
# or a code ref called with the name
sub _keeper {
    my ($keep) = @_;
    return sub { 0 } unless $keep;
    return $keep if ref $keep eq 'CODE';
    my %k;
    for (@$keep) { (my $base = $_) =~ s/\.(?:kvb|rst)\z//; $k{$base} = 1 }
    return sub { (my $base = $_[0]) =~ s/\.(?:kvb|rst)\z//; $k{$base} };
}

# Least recently used first; files used in the same second: the older chain position does not matter, by name
sub _lru { sort { $a->{mtime} <=> $b->{mtime} || $a->{name} cmp $b->{name} } @_ }

# store_evict($dir, max_bytes => $n, keep => [...] | sub {...}): delete least recently used files until the store
# holds at most max_bytes. Returns the names deleted.
sub store_evict {
    my ($dir, %o) = @_;
    croak "ROCm::Inference::Store::store_evict: max_bytes required" unless defined $o{max_bytes};
    my $keep = _keeper($o{keep});
    my @f = _lru(store_files($dir));
    my $total = 0;
    $total += $_->{bytes} for @f;
    my @gone;
    for my $f (@f) {
        last if $total <= $o{max_bytes};
        next if $keep->($f->{name});
        next unless unlink $f->{path};
        $total -= $f->{bytes};
        push @gone, $f->{name};
    }
    return @gone;
}

# Copy src to dst without ever leaving an incomplete dst visible: a temporary name in dst's directory, then rename
sub _copy_atomic {
    my ($src, $dst) = @_;
    my $tmp = "$dst.tmp.$$";
    File::Copy::copy($src, $tmp) or do { unlink $tmp; return 0 };
    rename($tmp, $dst) or do { unlink $tmp; return 0 };
    return 1;
}

# store_spill($hot, $cold, max_bytes => $n, keep => ...): as store_evict, but the files move to $cold (e.g. on
# disk) instead of being deleted. Returns the names moved.
sub store_spill {
    my ($hot, $cold, %o) = @_;
    croak "ROCm::Inference::Store::store_spill: max_bytes required" unless defined $o{max_bytes};
    croak "ROCm::Inference::Store::store_spill: $cold is not a directory" unless -d $cold;
    my $keep = _keeper($o{keep});
    my @f = _lru(store_files($hot));
    my $total = 0;
    $total += $_->{bytes} for @f;
    my @moved;
    for my $f (@f) {
        last if $total <= $o{max_bytes};
        next if $keep->($f->{name});
        next unless _copy_atomic($f->{path}, File::Spec->catfile($cold, $f->{name}));
        next unless unlink $f->{path};
        $total -= $f->{bytes};
        push @moved, $f->{name};
    }
    return @moved;
}

# store_fetch($cold, $hot, @names): bring files back into the store (atomically, so a running model never maps
# a partial file). A block's name also fetches its .rst if there is one. Returns the names fetched.
sub store_fetch {
    my ($cold, $hot, @names) = @_;
    my (%seen, @got);
    for my $name (@names) {
        (my $base = $name) =~ s/\.(?:kvb|rst)\z//;
        for my $n ("$base.kvb", "$base.rst") {
            next if $seen{$n}++;
            my $src = File::Spec->catfile($cold, $n);
            next unless -f $src;
            my $dst = File::Spec->catfile($hot, $n);
            next if -f $dst;
            push @got, $n if _copy_atomic($src, $dst);
        }
    }
    return @got;
}

1;

__END__

=head1 NAME

ROCm::Inference::Store - eviction and tiering for the ROCm::Inference KV block store

=head1 SYNOPSIS

    use ROCm::Inference qw(:all);
    use ROCm::Inference::Store qw(:all);

    store_open($m, '/dev/shm/kvstore', 4);
    ...
    # keep the hot store under 16 GB: least recently used blocks go to disk, the system prompt stays
    my @keep = store_block_names($m, $system_prompt_ids);
    store_spill('/dev/shm/kvstore', '/var/cache/kvstore', max_bytes => 16 << 30, keep => \@keep);
    # and the disk tier under 200 GB
    store_evict('/var/cache/kvstore', max_bytes => 200 << 30);
    # before resuming a conversation whose blocks went cold
    store_fetch('/var/cache/kvstore', '/dev/shm/kvstore', store_block_names($m, $conversation_ids));

=head1 DESCRIPTION

Plain functions over the store directory; the policy (budgets, what to keep, when to run) is the caller's.
The library writes C<< <hash>.kvb >> (KV blocks) and C<< <hash>.rst >> (state snapshots) and updates their
mtime when it uses them. Deleting or moving files is always safe: a missing file only means that part of a
prompt is prefilled again.

=over

=item store_files($dir), store_usage($dir)

The store's files (C<name>, C<path>, C<bytes>, C<mtime>); total bytes and count.

=item store_evict($dir, max_bytes => $n [, keep => \@names | sub { ... }])

Delete least recently used files until the directory holds at most C<$n> bytes. A kept block's
snapshot is kept too.

=item store_spill($hot, $cold, max_bytes => $n [, keep => ...])

The same, moving the files to C<$cold> instead.

=item store_fetch($cold, $hot, @names)

Copy files back (with their snapshots), atomically.

=back

=cut
