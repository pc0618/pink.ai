<?php
	$count = count($_FILES['upload']['name']);
	
	$files = array();
	$tags = "";

	for ($i = 0; $i < $count; $i++) {
		move_uploaded_file ($_FILES['upload']['tmp_name'][$i], "/var/www/html/data/image-" . $i . ".png");
		array_push($files, "-f data/image-" . $i . ".png");
		$tags = $tags . "-f data/image-" . $i . ".png ";
	}

	echo(shell_exec("python3 predict.py " . $tags));
?>
