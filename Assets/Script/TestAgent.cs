using UnityEngine;
using UnityEngine.InputSystem;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;

[RequireComponent(typeof(BehaviorParameters))]
[RequireComponent(typeof(DecisionRequester))]
public class TestAgent : Agent
{
    [Header("References")]
    [SerializeField] private Transform platformTransform;
    [SerializeField] private Rigidbody ballRigidbody;

    [Header("Platform Control")]
    [SerializeField] private float maxTiltAngle = 12f;
    [SerializeField] private float tiltSpeed = 120f;

    [Header("Episode Settings")]
    [SerializeField] private float spawnRadiusRatio = 0.35f;
    [SerializeField] private float resetHeightOffset = 0.08f;
    [SerializeField] private float fallHeightMargin = 0.35f;
    [SerializeField] private int trainingMaxStep = 5000;

    [Header("Rewards")]
    [SerializeField] private float surviveReward = 0.0025f;
    [SerializeField] private float centeredReward = 0.0035f;
    [SerializeField] private float fallPenalty = -1f;
    [SerializeField] private float outOfBoundsPenalty = -0.75f;

    private const int ObservationSize = 8;
    private const int ContinuousActionSize = 2;

    private BehaviorParameters behaviorParameters;
    private DecisionRequester decisionRequester;
    private BoxCollider platformCollider;
    private Rigidbody platformRigidbody;

    private Vector3 initialPlatformPosition;
    private Quaternion initialPlatformRotation;
    private float platformHalfExtentX;
    private float platformHalfExtentZ;
    private float trainingBoundsX;
    private float trainingBoundsZ;
    private float ballRadius = 0.1f;

    public override void Initialize()
    {
        platformTransform = platformTransform == null ? transform : platformTransform;

        behaviorParameters = GetComponent<BehaviorParameters>();
        decisionRequester = GetComponent<DecisionRequester>();
        platformCollider = platformTransform.GetComponent<BoxCollider>();
        platformRigidbody = platformTransform.GetComponent<Rigidbody>();

        ApplyMlAgentsConfiguration();
        CachePlatformState();
        EnsureBallReference();

        MaxStep = trainingMaxStep;
    }

    public override void OnEpisodeBegin()
    {
        ResetPlatform();
        ResetBall();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        EnsureBallReference();

        Vector3 localBallPosition = platformTransform.InverseTransformPoint(ballRigidbody.transform.position);
        Vector3 localBallVelocity = platformTransform.InverseTransformDirection(ballRigidbody.linearVelocity);
        Vector3 euler = platformTransform.localEulerAngles;

        float tiltX = NormalizeAngle(euler.x) / maxTiltAngle;
        float tiltZ = NormalizeAngle(euler.z) / maxTiltAngle;

        sensor.AddObservation(Mathf.Clamp(localBallPosition.x / Mathf.Max(trainingBoundsX, 0.001f), -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(localBallPosition.z / Mathf.Max(trainingBoundsZ, 0.001f), -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(localBallVelocity.x / 5f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(localBallVelocity.z / 5f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(localBallVelocity.y / 5f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(tiltX, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(tiltZ, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(localBallPosition.magnitude / Mathf.Max(Mathf.Max(trainingBoundsX, trainingBoundsZ), 0.001f), 0f, 1f));
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (actions.ContinuousActions.Length < ContinuousActionSize)
        {
            return;
        }

        EnsureBallReference();

        float actionX = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float actionZ = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        float targetTiltX = actionZ * maxTiltAngle;
        float targetTiltZ = -actionX * maxTiltAngle;
        Quaternion targetRotation = Quaternion.Euler(targetTiltX, initialPlatformRotation.eulerAngles.y, targetTiltZ);

        platformTransform.rotation = Quaternion.RotateTowards(
            platformTransform.rotation,
            targetRotation,
            tiltSpeed * Time.fixedDeltaTime);

        Vector3 localBallPosition = platformTransform.InverseTransformPoint(ballRigidbody.transform.position);
        float centered01 = 1f - Mathf.Clamp01(
            new Vector2(
                localBallPosition.x / Mathf.Max(trainingBoundsX, 0.001f),
                localBallPosition.z / Mathf.Max(trainingBoundsZ, 0.001f)).magnitude);

        AddReward(surviveReward);
        AddReward(centered01 * centeredReward);

        bool outOfPlatform =
            Mathf.Abs(localBallPosition.x) > trainingBoundsX ||
            Mathf.Abs(localBallPosition.z) > trainingBoundsZ;

        bool fellBelowPlatform =
            ballRigidbody.transform.position.y < initialPlatformPosition.y - fallHeightMargin;

        if (fellBelowPlatform)
        {
            AddReward(fallPenalty);
            EndEpisode();
            return;
        }

        if (outOfPlatform)
        {
            AddReward(outOfBoundsPenalty);
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;
        if (continuousActions.Length < ContinuousActionSize)
        {
            return;
        }

        float horizontal = 0f;
        float vertical = 0f;

        if (Keyboard.current != null)
        {
            if (Keyboard.current.leftArrowKey.isPressed || Keyboard.current.aKey.isPressed) horizontal -= 1f;
            if (Keyboard.current.rightArrowKey.isPressed || Keyboard.current.dKey.isPressed) horizontal += 1f;
            if (Keyboard.current.downArrowKey.isPressed || Keyboard.current.sKey.isPressed) vertical -= 1f;
            if (Keyboard.current.upArrowKey.isPressed || Keyboard.current.wKey.isPressed) vertical += 1f;
        }
        else
        {
            horizontal = Input.GetAxisRaw("Horizontal");
            vertical = Input.GetAxisRaw("Vertical");
        }

        continuousActions[0] = horizontal;
        continuousActions[1] = vertical;
    }

    private void ApplyMlAgentsConfiguration()
    {
        behaviorParameters.BehaviorName = "test_agent";
        behaviorParameters.BrainParameters.VectorObservationSize = ObservationSize;
        behaviorParameters.BrainParameters.NumStackedVectorObservations = 1;
        behaviorParameters.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(ContinuousActionSize);

        decisionRequester.DecisionPeriod = 1;
        decisionRequester.DecisionStep = 0;
        decisionRequester.TakeActionsBetweenDecisions = true;
    }

    private void CachePlatformState()
    {
        initialPlatformPosition = platformTransform.position;
        initialPlatformRotation = platformTransform.rotation;

        if (platformCollider == null)
        {
            platformCollider = platformTransform.gameObject.AddComponent<BoxCollider>();
        }

        Vector3 lossyScale = platformTransform.lossyScale;
        Vector3 scaledSize = Vector3.Scale(platformCollider.size, lossyScale);

        platformHalfExtentX = Mathf.Abs(scaledSize.x) * 0.5f;
        platformHalfExtentZ = Mathf.Abs(scaledSize.z) * 0.5f;
    }

    private void EnsureBallReference()
    {
        if (ballRigidbody != null)
        {
            UpdateTrainingBounds();
            return;
        }

        Rigidbody[] rigidbodies = FindObjectsByType<Rigidbody>(FindObjectsSortMode.None);
        foreach (Rigidbody candidate in rigidbodies)
        {
            if (candidate == platformRigidbody)
            {
                continue;
            }

            ballRigidbody = candidate;
            break;
        }

        if (ballRigidbody == null)
        {
            CreateTrainingBall();
        }

        UpdateTrainingBounds();
    }

    private void CreateTrainingBall()
    {
        GameObject ballObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        ballObject.name = "TrainingBall";
        ballObject.transform.localScale = Vector3.one * 0.2f;

        ballRigidbody = ballObject.AddComponent<Rigidbody>();
        ballRigidbody.mass = 0.5f;
        ballRigidbody.linearDamping = 0.1f;
        ballRigidbody.angularDamping = 0.05f;
        ballRigidbody.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
        ballRigidbody.interpolation = RigidbodyInterpolation.Interpolate;

        SphereCollider sphereCollider = ballObject.GetComponent<SphereCollider>();
        ballRadius = sphereCollider.radius * ballObject.transform.localScale.x;
    }

    private void UpdateTrainingBounds()
    {
        Collider ballCollider = ballRigidbody.GetComponent<Collider>();
        if (ballCollider != null)
        {
            ballRadius = Mathf.Max(ballCollider.bounds.extents.x, 0.05f);
        }

        trainingBoundsX = Mathf.Max(platformHalfExtentX - (ballRadius * 1.1f), 0.05f);
        trainingBoundsZ = Mathf.Max(platformHalfExtentZ - (ballRadius * 1.1f), 0.05f);
    }

    private void ResetPlatform()
    {
        platformTransform.SetPositionAndRotation(initialPlatformPosition, initialPlatformRotation);

        if (platformRigidbody != null)
        {
            platformRigidbody.linearVelocity = Vector3.zero;
            platformRigidbody.angularVelocity = Vector3.zero;
        }
    }

    private void ResetBall()
    {
        EnsureBallReference();

        Vector2 randomOffset = Random.insideUnitCircle * Mathf.Min(trainingBoundsX, trainingBoundsZ) * spawnRadiusRatio;
        Vector3 localSpawn = new Vector3(randomOffset.x, resetHeightOffset + ballRadius, randomOffset.y);
        Vector3 worldSpawn = platformTransform.TransformPoint(localSpawn);

        ballRigidbody.transform.SetPositionAndRotation(worldSpawn, Quaternion.identity);
        ballRigidbody.linearVelocity = Vector3.zero;
        ballRigidbody.angularVelocity = Vector3.zero;
    }

    private static float NormalizeAngle(float angle)
    {
        if (angle > 180f)
        {
            angle -= 360f;
        }

        return angle;
    }
}
