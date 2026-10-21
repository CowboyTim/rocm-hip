use strict;
use warnings;
use Test::More;

# Author test: every exported function is documented. The C-API wrappers documented as a group ("As the C functions
# of the same name") are listed there by name, which Pod::Coverage counts.
eval { require Test::Pod::Coverage; Test::Pod::Coverage->VERSION(1.08); 1 }
    or plan skip_all => 'Test::Pod::Coverage 1.08 required';
Test::Pod::Coverage::all_pod_coverage_ok({ also_private => [qr/^_/] });
