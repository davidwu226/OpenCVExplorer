package org.tacobots.opencvexplorer;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import android.content.Context;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCamera2View;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import fi.iki.elonen.NanoHTTPD;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "OpenCVExplorer";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);

    private CameraBridgeViewBase cameraView;

    private File cascadeFile;

    private int frameWidth;
    private int frameHeight;

    private BaseLoaderCallback loaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalfacel);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        cascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(cascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        cascadeClassifier = new CascadeClassifier(cascadeFile.getAbsolutePath());
                        if (cascadeClassifier.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                        }
                        cascadeDir.delete();

                    } catch (IOException e) {
                        Log.e(TAG, "I/O failed to load cascade classifier");
                        e.printStackTrace();
                    }

                    cameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    Mat rgba;
    Mat rgbaF;
    Mat rgbaT;

    private CascadeClassifier cascadeClassifier;

    private GripPipeline pipeline;

    private int lastTouchX;
    private int lastTouchY;

    private HTTPServer httpServer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.show_camera);

        cameraView = (JavaCameraView) findViewById(R.id.show_camera_activity_java_surface_view);
        cameraView.setVisibility(SurfaceView.VISIBLE);
        cameraView.setCvCameraViewListener(this);
        cameraView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                float x = event.getX() - ((v.getWidth() - frameWidth) / 2);
                float y = + event.getY() - ((v.getHeight() - frameHeight) / 2);
                Log.e(TAG, "Touch: "+(event.getX()-v.getLeft())+" "+(event.getY()-v.getTop()));
                Log.e(TAG, "Touch left/top: "+(v.getLeft())+" "+(v.getTop()));
                Log.e(TAG, "Touch width/height: "+(v.getWidth())+" "+(v.getHeight()));
                Log.e(TAG, "Touch x/y: "+x+" "+y);
                lastTouchX = Math.round(x);
                lastTouchY = Math.round(y);
                return true;
            };
        });
    }

    @Override
    public void onPause() {
        super.onPause();
        if (cameraView != null) {
            cameraView.disableView();
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, loaderCallback);
        } else {
            loaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        pipeline = new GripPipeline();
        httpServer = new HTTPServer();
        try {
            httpServer.start();
        } catch (IOException e) {
            Log.e(TAG, "Got exception "+e);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (cameraView != null) {
            cameraView.disableView();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        frameWidth = width;
        frameHeight = height;
        rgba = new Mat(height, width, CvType.CV_8UC4);
        rgbaF = new Mat(height, width, CvType.CV_8UC4);
        rgbaT = new Mat(height, width, CvType.CV_8UC4);
        Log.d(TAG, "Width "+width+" Height "+height);
        Log.d(TAG, "Library is "+Core.NATIVE_LIBRARY_NAME);
    }

    @Override
    public void onCameraViewStopped() {
        rgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        /*
        rgba = inputFrame.rgba();
        Core.transpose(rgba, rgbaT);
        Imgproc.resize(rgbaT, rgbaF, rgbaF.size(), 0,0, 0);
        Core.flip(rgbaF, rgba, 1);

        MatOfRect faces = new MatOfRect();

        Mat gray = inputFrame.gray();
        Mat grayF = new Mat(gray.cols(), gray.rows(), gray.type());
        Mat grayT = new Mat(gray.cols(), gray.rows(), gray.type());
        Core.transpose(gray, grayT);
        Imgproc.resize(grayT, grayF, grayF.size(), 0, 0, 0);
        Core.flip(grayF, gray, 1);

        long faceSize = Math.round(gray.rows() * 0.2);
        cascadeClassifier.detectMultiScale(gray, faces, 1.1, 2, 2, new Size(faceSize, faceSize), new Size());

        Rect[] facesArray = faces.toArray();

        for (int i = 0; i < facesArray.length; ++i) {
            Imgproc.rectangle(rgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
        }
        return rgba;*/

        /*
        Mat rgba = inputFrame.rgba();
        Core.transpose(rgba, rgba);
        Core.flip(rgba, rgba, 1);
        Mat rgb = new Mat();
        Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2BGR);
        pipeline.process(rgb);
        Features2d.drawKeypoints(rgb, pipeline.findBlobsOutput(), rgb, new Scalar(255, 0, 0), Features2d.DRAW_RICH_KEYPOINTS);
        Imgproc.cvtColor(rgb, rgba, Imgproc.COLOR_BGR2RGBA);
        return rgba;
        */

        Mat rgba = inputFrame.rgba();
        Core.transpose(rgba, rgba);
        Core.flip(rgba, rgba, 1);
        Mat rgb = new Mat();
        Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2BGR);
        pipeline.process(rgb);
        Imgproc.drawContours(rgba, pipeline.filterContoursOutput(),-1, new Scalar(0, 255, 0), 3);
        byte [] colors = new byte[12];
        rgba.get(lastTouchY, lastTouchX, colors);
        //Log.d(TAG, "Channels "+rgba.channels()+" Size "+rgba.elemSize());
        //Log.d(TAG, "Color at "+lastTouchX+" "+lastTouchY+" "+(colors[0] & 0xff)+" "+(colors[1] & 0xff)+" "+(colors[2] & 0xff));
        Mat rgbColor = new Mat(1, 1, CvType.CV_8UC3, new Scalar(255, 255, 0));
        Mat hsvColor = new Mat();
        Imgproc.cvtColor(rgbColor, hsvColor, Imgproc.COLOR_RGB2HSV);
        hsvColor.get(0, 0, colors);
        Log.d(TAG, "Yellow is "+(colors[0] & 0xff)+" "+(colors[1] & 0xff)+" "+(colors[2] & 0xff));

        jpgMat = new MatOfByte();
        Imgcodecs.imencode(".jpg", rgb, jpgMat);
        return rgba;
    }

    private MatOfByte jpgMat;

    public class HTTPServer extends NanoHTTPD {
        public HTTPServer() {
            super(9000);
        }

        @Override
        public Response serve(IHTTPSession session) {
            ByteArrayInputStream bis = new ByteArrayInputStream(jpgMat.toArray());
            return newFixedLengthResponse(Response.Status.OK, "image/jpeg", bis, bis.available());
        }
    }
}
