use strict;
use warnings;
use Test::More;

# Author test: the POD of every module parses
eval { require Test::Pod; Test::Pod->VERSION(1.22); 1 } or plan skip_all => 'Test::Pod 1.22 required';
Test::Pod::all_pod_files_ok();
