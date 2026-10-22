use strict;
use warnings;
use Test::More;

# Author test: every exported function is documented
eval { require Test::Pod::Coverage; Test::Pod::Coverage->VERSION(1.08); 1 }
    or plan skip_all => 'Test::Pod::Coverage 1.08 required';
Test::Pod::Coverage::all_pod_coverage_ok({ also_private => [qr/^_/] });
