use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir);

use ROCm::Inference::Store qw(:all);

# File policy only: no GPU or model needed
my $hot = tempdir(CLEANUP => 1);
my $cold = tempdir(CLEANUP => 1);
my $now = time;
my @names = map { sprintf('%064x', $_) } 1 .. 6;
sub mk {
    my ($dir, $name, $bytes, $age) = @_;
    open my $fh, '>', "$dir/$name" or die $!;
    print $fh 'x' x $bytes;
    close $fh;
    utime $now - $age, $now - $age, "$dir/$name";
}
mk($hot, "$names[$_].kvb", 100, 60 - 10 * $_) for 0 .. 5;  # block 0 the least recently used
mk($hot, "$names[1].rst", 500, 45);  # used after its block
mk($hot, "not-a-store-file.txt", 9999, 1000);

my ($bytes, $n) = store_usage($hot);
is($bytes, 1100, 'store_usage counts .kvb and .rst only');
is($n, 7, '7 store files');

my @gone = store_evict($hot, max_bytes => 800, keep => ["$names[0].kvb"]);
is_deeply(\@gone, ["$names[1].kvb", "$names[1].rst"], 'LRU eviction skips the kept block, deletes the next oldest');
ok(-f "$hot/$names[0].kvb" && -f "$hot/not-a-store-file.txt", 'kept block and unrelated files remain');
is(scalar store_usage($hot), 500, 'now 500 bytes');

my @moved = store_spill($hot, $cold, max_bytes => 300);
is_deeply(\@moved, ["$names[0].kvb", "$names[2].kvb"], 'spill moves the least recently used to the cold tier');
ok(!-e "$hot/$names[0].kvb" && -s "$cold/$names[0].kvb" == 100, 'moved, with its contents');

my @got = store_fetch($cold, $hot, "$names[2].kvb");
is_deeply(\@got, ["$names[2].kvb"], 'fetch brings a block back');
ok(-s "$hot/$names[2].kvb" == 100 && !glob("$hot/*.tmp.*"), 'complete, no temporary file left');

done_testing;
