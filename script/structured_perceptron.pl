use strict;
use warnings;
use utf8;
use Encode;
use List::Util;
use List::MoreUtils;
use List::Rubyish;
use Clone qw(clone);

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

sub get_pos_labels {
    my $data       = shift;
    my $pos_labels = List::Rubyish->new;
    $data->inject(
        List::Rubyish->new,
        sub {
            my $result = $_[0];
            $_->each( sub { $result->push( $_->{pos} ); } );
            $result;
        }
    )->uniq;
}

sub extract_features {
    my ( $sentence, $index, $pos_prev, $pos_next ) = @_;
    my $features = List::Rubyish->new;
    my $w = $index < $sentence->size ? $sentence->[$index]->{w} : "EOS";
    my $w_prev = $index - 1 >= 0 ? $sentence->[$index - 1]->{w} : "";
    my $w_next = $index + 1 < $sentence->size ? $sentence->[$index + 1]->{w} : "";
    $features->push( "transition_feature:" . $pos_prev . "+" . $pos_next );
    $features->push( "emission_feature:" . $pos_next . "+" . $w );
    $features->push( "emission_feature_prev:" . $pos_next . "+" . $w_prev );
    $features->push( "emission_feature_next:" . $pos_next . "+" . $w_next );
    $features;
}

sub inner_product {
    my ( $features, $weight ) = @_;
    my $score = 0.0;
    $features->inject( 0.0, sub { $_[0] + ( $weight->{ $_[1] } || 0.0 ); } );
}

sub argmax {
    my ( $sentence_, $weight, $pos_labels ) = @_;
    my $sentence = clone $sentence_;
    my $best_edge = forward_step( $sentence, $weight, $pos_labels );
    backward_step( $sentence, $best_edge );
}

sub backward_step {
    my ( $sentence, $best_edge ) = @_;
    my $max_idx   = $sentence->size + 1;
    my $next_edge = $best_edge->{"$max_idx EOS"};
    while ( $next_edge ne "0 BOS" ) {
        die "Cannot backtrack" unless $next_edge;
        my ( $idx, $pos ) = split /\s/, $next_edge;
        $sentence->[ $idx - 1 ]->{pos} = $pos;
        $next_edge = $best_edge->{"$next_edge"};
    }
    $sentence;
}

sub forward_step {
    my ( $sentence, $weight, $pos_labels ) = @_;
    my $best_score = {};
    my $best_edge  = {};
    my $bos        = "BOS";
    $best_score->{"0 $bos"} = 0.0;
    $best_edge->{"0 $bos"}  = undef;
    for my $w_idx ( 0 .. $sentence->size - 1 ) {
        foreach my $prev ( $w_idx == 0 ? ($bos) : @$pos_labels ) {
            foreach my $next (@$pos_labels) {
                my $features = extract_features( $sentence, $w_idx, $prev, $next );
                my $cum_score = $w_idx == 0 ? $best_score->{"$w_idx $bos"} : ( $best_score->{"$w_idx $prev"} || 0.0 ) ;
                my $score = $cum_score + inner_product( $features, $weight );
                my $w_next_idx = $w_idx + 1;
                if ( ( not defined $best_score->{"$w_next_idx $next"} ) ||
                     ( $best_score->{"$w_next_idx $next"} <= $score ) ) {
                    $best_score->{"$w_next_idx $next"} = $score;
                    $best_edge->{"$w_next_idx $next"}  = "$w_idx $prev";
                }
            }
        }
    }
    my $eos = "EOS";
    my $w_idx = $sentence->size;
    for my $prev (@$pos_labels) {
        my $next = $eos;
        my $features = extract_features( $sentence, $w_idx, $prev, $next );
        my $score = $best_score->{"$w_idx $prev"} + inner_product( $features, $weight );
        my $w_next_idx = $w_idx + 1;
        if ( ( not defined $best_score->{"$w_next_idx $next"} ) || ( $best_score->{"$w_next_idx $next"} <= $score ) ) {
            $best_score->{"$w_next_idx $next"} = $score;
            $best_edge->{"$w_next_idx $next"}  = "$w_idx $prev";
        }
    }
    $best_edge;
}

sub get_features {
    my $sentence = shift;
    my $gold_features = List::Rubyish->new;
    for my $index ( 0 .. $sentence->size - 1 ) {
        my $prev_pos = $sentence->[ $index - 1 ]->{pos} || "BOS";
        my $next_pos = $sentence->[$index]->{pos};
        my $features = extract_features( $sentence, $index, $prev_pos, $next_pos );
        $gold_features->concat($features);
    }
    return $gold_features;
}

sub learn {
    my ( $weight, $cum_weight, $sentence, $predict_sentence, $n, $pos_labels ) = @_;
    my $gold_features = get_features($sentence);
    my $predict_features = get_features($predict_sentence);

    # update weight
    for my $feature (@$gold_features) {
        $weight->{$feature} += 1;
        $cum_weight->{$feature} += $n;
    }
    for my $feature (@$predict_features) {
        $weight->{$feature} -= 1;
        $cum_weight->{$feature} -= $n;
    }
}

sub get_final_weight {
    my ( $weight, $cum_weight, $n ) = @_;
    my $final_weight = { %$weight };
    while ( my ( $k, $v ) = each(%$cum_weight) ) {
        $final_weight->{$k} -= $v / $n;
    }
    $final_weight;
}

sub accuracy {
    my ($golds, $predicts) = @_;
    my $correct = 0.0;
    my $num = 0.0;
    for my $index ( 0 .. $golds->size - 1 ) {
        my $gold_pos_labels = $golds->[$index]->map(sub {$_->{pos}});
        my $predict_pos_labels = $predicts->[$index]->map(sub {$_->{pos}});
        for my $i ( 0 .. $gold_pos_labels->size - 1 ) {
            $correct++ if $gold_pos_labels->[$i] eq $predict_pos_labels->[$i];
            $num++;
        }
    }
    $correct / $num;
}

sub pos_labels_str {
    my $sentence = shift;
    join( ", ", @{ $sentence->map( sub { $_->{pos} } ) } );
}

my $pos_filename_train = "/Users/yasuhisa/.ghq/github.com/neubig/nlptutorial/data/wiki-ja-train.word_pos";
my $train_data = read_data($pos_filename_train);
my $pos_labels = get_pos_labels $train_data;

my $pos_filename_test = "/Users/yasuhisa/.ghq/github.com/neubig/nlptutorial/data/wiki-ja-test.word_pos";
my $test_data = read_data($pos_filename_test);

my $weight = {};
my $cum_weight = {};
my $n = 1;
for my $iter ( 0 .. 10 ) {
    print "Iter: $iter\n";
    for my $gold ( List::Util::shuffle @$train_data) {
        my $predict = argmax($gold, $weight, $pos_labels);
        # 正解と一致したら重みベクトルは更新しない
        if ( pos_labels_str($gold) ne pos_labels_str($predict) ) {
            learn( $weight, $cum_weight, $gold, $predict, $n, $pos_labels );
            $n++;
        }
    }
    my $w = get_final_weight( $weight, $cum_weight, $n );
    my $predicts = List::Rubyish->new;
    for my $gold (@$test_data) {
        my $predict = argmax( $gold, $w, $pos_labels );
        $predicts->push($predict);
    }
    print accuracy( $test_data, $predicts ), "\n";
}
