<?php
require_once ("class_neuralnetwork.php");

// Create a new neural network with 3 input neurons,
// 4 hidden neurons, and 1 output neuron
$n = new NeuralNetwork(3, 4, 1);
$n->setVerbose(false);

// Add test-data to the network. In this case,
// we want the network to learn the 'XOR'-function
$n->addTestData(array (-1, -1, 1), array (-1));
$n->addTestData(array (-1,  1, 1), array ( 1));
$n->addTestData(array ( 1, -1, 1), array ( 1));
$n->addTestData(array ( 1,  1, 1), array (-1));

// we try training the network for at most $max times
$max = 3;
$i = 0;

echo "Learning the XOR function".PHP_EOL;
// train the network in max 1000 epochs, with a max squared error of 0.01
while (!($success = $n->train(1000, 0.01)) && ++$i<$max) {
	echo "Round $i: No success...".PHP_EOL;
}

// print a message if the network was succesfully trained
if ($success) {
    $epochs = $n->getEpoch();
	echo "Success in $epochs training rounds!".PHP_EOL;
}

echo "Result".PHP_EOL;

// in any case, we print the output of the neural network
for ($i = 0; $i < count($n->trainInputs); $i ++) {
	$output = $n->calculate($n->trainInputs[$i]);
	echo "Testset $i; ".PHP_EOL;
	echo "expected output = (".implode(", ", $n->trainOutput[$i]).") ";
	echo "output from neural network = (".implode(", ", $output).")".PHP_EOL;
}

//echo "<h2>Internal network state</h2>";
//$n->showWeights($force=true);

// Now, play around with some of the network's parameters a bit, to see how it 
// influences the result
$learningRates = array(0.1, 0.25, 0.5, 0.75, 1);
$momentum = array(0.2, 0.4, 0.6, 0.8, 1);
$rounds = array(100, 500, 1000, 2000);
$errors = array(0.1, 0.05, 0.01, 0.001);

echo "Playing around...".PHP_EOL;
echo "The following is to show how changing the momentum & learning rate, 
in combination with the number of rounds and the maximum allowable error, can 
lead to wildly differing results. To obtain the best results for your 
situation, play around with these numbers until you find the one that works
best for you.".PHP_EOL;
echo "The values displayed here are chosen randomly, so you can reload 
the page to see another set of values...".PHP_EOL;

for ($j=0; $j<10; $j++) {
	// no time-outs
	set_time_limit(0);
	
	$lr = $learningRates[array_rand($learningRates)];
	$m = $momentum[array_rand($momentum)];
	$r = $rounds[array_rand($rounds)];
	$e = $errors[array_rand($errors)];
	echo "Learning rate $lr, momentum $m @ ($r rounds, max sq. error $e)".PHP_EOL;
	$n->clear();
	$n->setLearningRate($lr);
	$n->setMomentum($m);
	$i = 0;
	while (!($success = $n->train($r, $e)) && ++$i<$max) {
		echo "Round $i: No success...".PHP_EOL;
		flush();
	}

	// print a message if the network was succesfully trained
	if ($success) {
	    $epochs = $n->getEpoch();
		echo "Success in $epochs training rounds!".PHP_EOL;

		
		for ($i = 0; $i < count($n->trainInputs); $i ++) {
			$output = $n->calculate($n->trainInputs[$i]);
			echo "Testset $i; ".PHP_EOL;
			echo "expected output = (".implode(", ", $n->trainOutput[$i]).") ".PHP_EOL;
			echo "output from neural network = (".implode(", ", $output).")".PHP_EOL;
		}
		
	}
}
?>