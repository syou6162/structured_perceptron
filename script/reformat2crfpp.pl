use strict;
use warnings;
use utf8;
use Encode;
use List::Rubyish;

sub parse_line {
    my $line     = shift;
    my $sentence = List::Rubyish->new;
    my $pairs    = [ split /\s/, decode_utf8 $line ];
    foreach my $pair (@$pairs) {
        my ( $w, $pos ) = split /_/, $pair;
        $sentence->push( { w => $w, pos => $pos } ) if $w && $pos;
    }
    $sentence;
}

sub read_data {
    my $filename = shift;
    my $FH;
    my $train_data = List::Rubyish->new;
    open $FH, "<:utf8", $filename;
    while (<$FH>) {
        my $line = $_;
        chomp $line;
        $train_data->push( parse_line($line) );
    }
    close $FH;
    $train_data;
}

my $pos_filename = $ARGV[0];
my $data = read_data($pos_filename);

foreach my $sentence ( @$data ) {
    foreach my $word ( @$sentence ) {
        print encode_utf8 $word->{w} . "\t" . $word->{pos}, "\n";
    }
    print "\n"
}
