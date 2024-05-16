<?php
$uploadDir = 'C:\xampp\htdocs\URasethetic\images\Capsule wardrobe management .zip';
$uploadedFiles = [];

if (!empty($_FILES['images']['name'][0])) {
    foreach ($_FILES['images']['tmp_name'] as $key => $tmp_name) {
        $file_name = $_FILES['images']['name'][$key];
        $file_tmp = $_FILES['images']['tmp_name'][$key];
        $file_type = $_FILES['images']['type'][$key];
        $file_size = $_FILES['images']['size'][$key];
        $file_ext = strtolower(end(explode('.', $file_name)));
        
        $extensions = array("jpeg","jpg","png");
        
        if (in_array($file_ext, $extensions) === false) {
            // Handle invalid file types
        }
        
        if ($file_size > 2097152) {
            // Handle files larger than 2MB
        }
        
        $uploadPath = $uploadDir . $file_name;
        
        if (move_uploaded_file($file_tmp, $uploadPath)) {
            $uploadedFiles[] = $uploadPath;
        } else {
            // Handle file upload failure
        }
    }
}
?>
