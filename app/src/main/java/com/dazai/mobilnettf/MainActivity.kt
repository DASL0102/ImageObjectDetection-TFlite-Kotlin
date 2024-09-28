package com.dazai.mobilnettf
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var textViewResult: TextView
    private lateinit var tflite: Interpreter
    private val IMAGE_CAPTURE_CODE = 1001
    private lateinit var bitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        textViewResult = findViewById(R.id.textViewResult)

        // Inicializar el modelo TensorFlow Lite
        tflite = Interpreter(loadModelFile("mobilenet_v1.tflite"))

        val buttonTakePhoto: Button = findViewById(R.id.buttonTakePhoto)
        buttonTakePhoto.setOnClickListener {
            val takePicture = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(takePicture, IMAGE_CAPTURE_CODE)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == IMAGE_CAPTURE_CODE && resultCode == Activity.RESULT_OK) {
            val extras = data?.extras
            bitmap = extras?.get("data") as Bitmap

            // Convertir el bitmap a ARGB_8888
            val convertedBitmap = convertToARGB8888(bitmap)

            imageView.setImageBitmap(convertedBitmap)
            classifyImage(convertedBitmap) // Usar el bitmap convertido
        }
    }


    private fun convertToARGB8888(bitmap: Bitmap): Bitmap {
        val outputBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(outputBitmap)
        canvas.drawBitmap(bitmap, 0f, 0f, null)
        return outputBitmap
    }


    private fun classifyImage(bitmap: Bitmap) {
        // Redimensionar la imagen
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        // Cargar el modelo TensorFlow Lite
        val model = Interpreter(loadModelFile("mobilenet_v1.tflite"))

        // Preparar la imagen para el modelo
        val tensorImage = TensorImage(DataType.UINT8)
        tensorImage.load(resizedBitmap) // Ahora debería estar en ARGB_8888

        // Crear un TensorBuffer para las salidas del modelo
        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 1001), DataType.UINT8)

        // Ejecutar el modelo
        model.run(tensorImage.buffer, outputBuffer.buffer.rewind())

        // Obtener etiquetas (labels) para la clasificación
        val labels = assets.open("labels_mobilenet.txt").bufferedReader().use { it.readLines() }

        // Convertir el resultado a un formato legible
        val labeledProbabilities = TensorLabel(labels, outputBuffer).mapWithFloatValue
        val maxLabel = labeledProbabilities.maxByOrNull { it.value }?.key ?: "Unknown"

        // Mostrar el resultado en la UI
        textViewResult.text = maxLabel

        // Cerrar el modelo
        model.close()
    }


   // para otros modelos ********************************
//    private fun classifyImage(bitmap: Bitmap, inputSize: Int, dataType: DataType) {
//        // Redimensionar la imagen
//        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
//
//        // Cargar el modelo TensorFlow Lite
//        val model = Interpreter(loadModelFile("your_model.tflite"))
//
//        // Preparar la imagen para el modelo
//        val tensorImage = TensorImage(dataType)
//        tensorImage.load(resizedBitmap)
//
//        // Crear un TensorBuffer para las salidas del modelo
//        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, outputSize), DataType.FLOAT32)
//
//        // Ejecutar el modelo
//        model.run(tensorImage.buffer, outputBuffer.buffer.rewind())
//
//        // Obtener etiquetas (labels) para la clasificación
//        val labels = assets.open("labels_your_model.txt").bufferedReader().use { it.readLines() }
//
//        // Convertir el resultado a un formato legible
//        val labeledProbabilities = TensorLabel(labels, outputBuffer).mapWithFloatValue
//        val maxLabel = labeledProbabilities.maxByOrNull { it.value }?.key ?: "Unknown"
//
//        // Mostrar el resultado en la UI
//        textViewResult.text = maxLabel
//
//        // Cerrar el modelo
//        model.close()
//    }




    override fun onDestroy() {
        super.onDestroy()
        // Cerrar el intérprete
        tflite.close()
    }

    // Función para cargar el archivo del modelo
    @Throws(IOException::class)
    private fun loadModelFile(modelFilename: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelFilename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
}
