using UnityEngine;

public class RealBallDetector : MonoBehaviour
{
    [Header("Inputs")]
    public Camera detectionCamera;
    public RenderTexture sensorTexture;

    public Transform plateOrigin;

    [Header("Outputs")]
    public float output_X;
    public float output_Y;
    public bool HasBall { get; private set; }

    private Texture2D readableTexture;
    private Rect readRect;

    void Start()
    {
        if (sensorTexture == null)
        {
            Debug.LogWarning("Sensor Texture is not set.", this);
            return;
        }

        readableTexture = new Texture2D(sensorTexture.width, sensorTexture.height);
        readRect = new Rect(0, 0, sensorTexture.width, sensorTexture.height);
    }

    void Update()
    {
        if (SystemInfo.graphicsDeviceType == UnityEngine.Rendering.GraphicsDeviceType.Null)
        {
            HasBall = false;
            return;
        }

        if (detectionCamera == null || sensorTexture == null || readableTexture == null)
        {
            HasBall = false;
            return;
        }

        RenderTexture.active = sensorTexture;
        readableTexture.ReadPixels(readRect, 0, 0);
        readableTexture.Apply();
        RenderTexture.active = null;
        Color[] pixels = readableTexture.GetPixels();

        float minX_px = sensorTexture.width;
        float maxX_px = 0;
        float minY_px = sensorTexture.height;
        float maxY_px = 0;
        bool ballFound = false;

        for (int y = 0; y < sensorTexture.height; y++)
        {
            for (int x = 0; x < sensorTexture.width; x++)
            {
                Color pixelColor = pixels[(y * sensorTexture.width) + x];
                if (pixelColor.r > 0.1f)
                {
                    ballFound = true;
                    if (x < minX_px) minX_px = x;
                    if (x > maxX_px) maxX_px = x;
                    if (y < minY_px) minY_px = y;
                    if (y > maxY_px) maxY_px = y;
                }
            }
        }

        if (ballFound)
        {
            HasBall = true;
            float pixelCenterX = (minX_px + maxX_px) / 2f;
            float pixelCenterY = (minY_px + maxY_px) / 2f;

            float viewportX = pixelCenterX / sensorTexture.width;
            float viewportY = pixelCenterY / sensorTexture.height;

            float z_distance = 10f;
            Vector3 worldPosition = detectionCamera.ViewportToWorldPoint(new Vector3(viewportX, viewportY, z_distance));


            if (plateOrigin != null)
            {
                Vector3 relativePosition = plateOrigin.InverseTransformPoint(worldPosition);

                output_X = relativePosition.x;
                output_Y = relativePosition.z;
            }
            else
            {
                //Debug.LogWarning("Plate Origin (Transform) is not set! Using world coordinates.");
                output_X = worldPosition.x;
                output_Y = worldPosition.z;
            }

            //Debug.Log($"[Detected] Relative Coords (X, Y): ({output_X:F3}, {output_Y:F3})");
        }
        else
        {
            HasBall = false;
            //Debug.Log("[No Ball Detected]");
        }
    }
}
