use strict;
use warnings;
use utf8;
use Encode;

my $num = 0;
my $correct = 0;
while (my $line = <STDIN>){
    chomp($line);
    next if $line eq '';
    my ($w, $gold, $predict) = split /\t/, $line;
    $num++;
    $correct++ if $gold eq $predict;
}

print $correct / $num, "\n";
