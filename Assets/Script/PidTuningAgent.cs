using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;

[RequireComponent(typeof(BehaviorParameters))]
[RequireComponent(typeof(DecisionRequester))]
public class PidTuningAgent : Agent
{
    [Header("Control Graph")]
    [SerializeField] private Controller controller;
    [SerializeField] private PIDController pidControllerX;
    [SerializeField] private PIDController pidControllerZ;
    [SerializeField] private RealBallDetector ballSensor;
    [SerializeField] private Transform plateTransform;

    [Header("Optional Reset References")]
    [SerializeField] private Rigidbody ballRigidbody;
    [SerializeField] private Transform ballTransform;

    [Header("Behavior")]
    [SerializeField] private string behaviorName = "pid_tuning_agent";
    [SerializeField] private int decisionPeriod = 5;
    [SerializeField] private int maxEpisodeSteps = 3000;
    [SerializeField] private bool applyPidOnlyAtEpisodeStart = true;

    [Header("Debug UI")]
    [SerializeField] private bool showPidOverlay = true;
    [SerializeField] private Vector2 overlayPosition = new Vector2(20f, 20f);
    [SerializeField] private int overlayFontSize = 24;
    [SerializeField] private Vector2 overlayPanelSize = new Vector2(900f, 120f);

    [Header("Scene Gizmos")]
    [SerializeField] private bool showResetGizmos = true;

    [Header("PID Gain Ranges")]
    [SerializeField] private Vector2 kpRange = new Vector2(0f, 10f);
    [SerializeField] private Vector2 kiRange = new Vector2(0f, 2f);
    [SerializeField] private Vector2 kdRange = new Vector2(0f, 2f);

    [Header("PID Quantization")]
    [SerializeField] private float kpStep = 0.5f;
    [SerializeField] private float kiStep = 0.01f;
    [SerializeField] private float kdStep = 0.05f;

    [Header("Reset")]
    [SerializeField] private float spawnRadius = 0.25f;
    [SerializeField] private float minimumSpawnRadius = 0.05f;
    [SerializeField] private float spawnHeightAbovePlate = 0.1f;
    [SerializeField] private float failDistance = 0.55f;
    [SerializeField] private float failHeightBelowPlate = 0.3f;

    [Header("Rewards")]
    [SerializeField] private float centeredRewardWeight = 0.015f;
    [SerializeField] private float lowVelocityRewardWeight = 0.003f;
    [SerializeField] private float velocityReference = 5f;
    [SerializeField] private float failurePenalty = -10f;

    private const int ObservationSize = 13;
    private const int ContinuousActionSize = 3;

    private BehaviorParameters behaviorParameters;
    private DecisionRequester decisionRequesterComponent;

    private Vector3 initialPlatePosition;
    private Quaternion initialPlateRotation;
    private Vector3 initialBallPosition;
    private Quaternion initialBallRotation;

    private bool hasAppliedPidThisEpisode;
    private bool pendingEpisodeDecision;
    private int episodeCounter;
    private bool episodeRewardBreakdownReported = true;
    private GUIStyle overlayBoxStyle;
    private GUIStyle overlayLabelStyle;
    private string lastEpisodeEndReason = "None";
    private float episodeCenteredRewardTotal;
    private float episodeLowVelocityRewardTotal;
    private float episodeFailurePenaltyTotal;
    private float lastRadialDistance;
    private float lastPlanarVelocity;
    private float lastBallWorldY;
    private float lastFailWorldYThreshold;

    public override void Initialize()
    {
        ResolveReferences();

        behaviorParameters = GetComponent<BehaviorParameters>();
        decisionRequesterComponent = GetComponent<DecisionRequester>();

        ApplyBehaviorConfiguration();

        if (plateTransform != null)
        {
            initialPlatePosition = plateTransform.position;
            initialPlateRotation = plateTransform.rotation;
        }

        if (ballTransform != null)
        {
            initialBallPosition = ballTransform.position;
            initialBallRotation = ballTransform.rotation;
        }

        MaxStep = maxEpisodeSteps;
        ResetPidState();
    }

    private void Reset()
    {
        ResolveReferences();
        ApplyBehaviorConfiguration();
    }

    private void OnValidate()
    {
        ResolveReferences();
        ApplyBehaviorConfiguration();
    }

    public override void OnEpisodeBegin()
    {
        ResolveReferences();
        ReportPendingEpisodeRewardBreakdown();
        ResetEpisodeRewardTracking();
        episodeCounter++;

        if (controller != null)
        {
            controller.targetPosition = Vector2.zero;
        }

        ResetPidState();
        ResetPlate();
        ResetBall();
        hasAppliedPidThisEpisode = false;
        pendingEpisodeDecision = true;
        lastEpisodeEndReason = "Running";
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        ResolveReferences();

        Vector3 localBallPosition = GetLocalBallPosition();
        Vector3 localBallVelocity = GetLocalBallVelocity();

        float plateAngleX = plateTransform != null ? NormalizeAngle(plateTransform.localEulerAngles.x) : 0f;
        float plateAngleZ = plateTransform != null ? NormalizeAngle(plateTransform.localEulerAngles.z) : 0f;

        Vector2 error = new Vector2(-localBallPosition.x, -localBallPosition.z);

        sensor.AddObservation(Mathf.Clamp(localBallPosition.x / failDistance, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(localBallPosition.z / failDistance, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(localBallVelocity.x / 5f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(localBallVelocity.z / 5f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(error.x / failDistance, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(error.y / failDistance, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(plateAngleX / 30f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(plateAngleZ / 30f, -1f, 1f));

        sensor.AddObservation(NormalizeGain(pidControllerX != null ? pidControllerX.Kp : 0f, kpRange));
        sensor.AddObservation(NormalizeGain(pidControllerX != null ? pidControllerX.Ki : 0f, kiRange));
        sensor.AddObservation(NormalizeGain(pidControllerX != null ? pidControllerX.Kd : 0f, kdRange));

        float radialDistance = new Vector2(localBallPosition.x, localBallPosition.z).magnitude;
        sensor.AddObservation(Mathf.Clamp(radialDistance / failDistance, 0f, 1f));
        sensor.AddObservation(Mathf.Clamp(localBallVelocity.magnitude / 5f, 0f, 1f));
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (actions.ContinuousActions.Length < ContinuousActionSize)
        {
            return;
        }

        ResolveReferences();

        if (!applyPidOnlyAtEpisodeStart || !hasAppliedPidThisEpisode)
        {
            ApplyPidActions(actions.ContinuousActions);
            hasAppliedPidThisEpisode = true;
            pendingEpisodeDecision = false;
            LogCurrentPidValues();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> continuous = actionsOut.ContinuousActions;
        if (continuous.Length < ContinuousActionSize)
        {
            return;
        }

        continuous[0] = NormalizeManualGain(pidControllerX != null ? pidControllerX.Kp : 0f, kpRange);
        continuous[1] = NormalizeManualGain(pidControllerX != null ? pidControllerX.Ki : 0f, kiRange);
        continuous[2] = NormalizeManualGain(pidControllerX != null ? pidControllerX.Kd : 0f, kdRange);
    }

    private void ApplyPidActions(ActionSegment<float> continuousActions)
    {
        if (pidControllerX == null || pidControllerZ == null)
        {
            return;
        }

        float sharedKp = QuantizeGain(
            ScaleAction(continuousActions[0], kpRange.x, kpRange.y),
            kpRange,
            kpStep);
        float sharedKi = QuantizeGain(
            ScaleAction(continuousActions[1], kiRange.x, kiRange.y),
            kiRange,
            kiStep);
        float sharedKd = QuantizeGain(
            ScaleAction(continuousActions[2], kdRange.x, kdRange.y),
            kdRange,
            kdStep);

        pidControllerX.Kp = sharedKp;
        pidControllerX.Ki = sharedKi;
        pidControllerX.Kd = sharedKd;

        pidControllerZ.Kp = sharedKp;
        pidControllerZ.Ki = sharedKi;
        pidControllerZ.Kd = sharedKd;
    }

    private bool HasFailed(float radialDistance, out string failReason)
    {
        failReason = string.Empty;

        if (radialDistance > failDistance)
        {
            failReason = $"Distance ({radialDistance:F3} > {failDistance:F3})";
            return true;
        }

        if (ballTransform != null && plateTransform != null)
        {
            float failYThreshold = plateTransform.position.y - failHeightBelowPlate;
            if (ballTransform.position.y < failYThreshold)
            {
                failReason = $"Height ({ballTransform.position.y:F3} < {failYThreshold:F3})";
                return true;
            }
        }

        return false;
    }

    private void ResetBall()
    {
        if (plateTransform == null || ballTransform == null)
        {
            return;
        }

        Vector2 offset = GetSpawnOffset();
        Vector3 spawnPosition = plateTransform.TransformPoint(
            new Vector3(offset.x, spawnHeightAbovePlate, offset.y));

        ballTransform.SetPositionAndRotation(spawnPosition, initialBallRotation);

        if (ballRigidbody != null)
        {
            ballRigidbody.linearVelocity = Vector3.zero;
            ballRigidbody.angularVelocity = Vector3.zero;
        }
    }

    private Vector2 GetSpawnOffset()
    {
        float outerRadius = Mathf.Max(spawnRadius, 0f);
        float innerRadius = Mathf.Clamp(minimumSpawnRadius, 0f, outerRadius);

        if (outerRadius <= 0f)
        {
            if (innerRadius <= 0f)
            {
                return Vector2.right * 0.05f;
            }

            float forcedAngle = Random.Range(0f, Mathf.PI * 2f);
            return new Vector2(Mathf.Cos(forcedAngle), Mathf.Sin(forcedAngle)) * innerRadius;
        }

        if (innerRadius >= outerRadius)
        {
            float ringAngle = Random.Range(0f, Mathf.PI * 2f);
            return new Vector2(Mathf.Cos(ringAngle), Mathf.Sin(ringAngle)) * outerRadius;
        }

        float angle = Random.Range(0f, Mathf.PI * 2f);
        float radius = Mathf.Sqrt(Random.Range(innerRadius * innerRadius, outerRadius * outerRadius));
        return new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)) * radius;
    }

    private void ResetPlate()
    {
        if (plateTransform == null)
        {
            return;
        }

        plateTransform.SetPositionAndRotation(initialPlatePosition, initialPlateRotation);
    }

    private void ResetPidState()
    {
        if (pidControllerX != null)
        {
            pidControllerX.ResetController();
        }

        if (pidControllerZ != null)
        {
            pidControllerZ.ResetController();
        }
    }

    private void FixedUpdate()
    {
        if (applyPidOnlyAtEpisodeStart && pendingEpisodeDecision)
        {
            RequestDecision();
            pendingEpisodeDecision = false;
        }

        if (!hasAppliedPidThisEpisode)
        {
            return;
        }

        Vector3 localBallPosition = GetLocalBallPosition();
        Vector3 localBallVelocity = GetLocalBallVelocity();
        float radialDistance = new Vector2(localBallPosition.x, localBallPosition.z).magnitude;
        lastRadialDistance = radialDistance;
        float planarVelocity = new Vector2(localBallVelocity.x, localBallVelocity.z).magnitude;
        lastPlanarVelocity = planarVelocity;
        lastBallWorldY = ballTransform != null ? ballTransform.position.y : 0f;
        lastFailWorldYThreshold = plateTransform != null ? plateTransform.position.y - failHeightBelowPlate : 0f;

        float safeFailDistance = Mathf.Max(failDistance, 0.0001f);
        float safeVelocityReference = Mathf.Max(velocityReference, 0.0001f);
        float normalizedDistance = Mathf.Clamp01(radialDistance / safeFailDistance);
        float normalizedVelocity = Mathf.Clamp01(planarVelocity / safeVelocityReference);

        float centeredStepReward = (1f - (normalizedDistance * normalizedDistance)) * centeredRewardWeight;
        float lowVelocityStepReward = (1f - normalizedVelocity) * lowVelocityRewardWeight;
        float stepReward = centeredStepReward + lowVelocityStepReward;

        AddTrackedReward(ref episodeCenteredRewardTotal, centeredStepReward);
        AddTrackedReward(ref episodeLowVelocityRewardTotal, lowVelocityStepReward);

        if (HasFailed(radialDistance, out string failReason))
        {
            AddTrackedReward(ref episodeFailurePenaltyTotal, failurePenalty);
            AddReward(failurePenalty);
            lastEpisodeEndReason = failReason;
            Debug.Log($"[PidTuningAgent] EndEpisode by {failReason}", this);
            ReportEpisodeRewardBreakdown();
            EndEpisode();
            return;
        }

        AddReward(stepReward);
    }

    private void AddTrackedReward(ref float accumulator, float rewardValue)
    {
        accumulator += rewardValue;
    }

    private void ResetEpisodeRewardTracking()
    {
        episodeCenteredRewardTotal = 0f;
        episodeLowVelocityRewardTotal = 0f;
        episodeFailurePenaltyTotal = 0f;
        episodeRewardBreakdownReported = false;
    }

    private void ReportPendingEpisodeRewardBreakdown()
    {
        if (!episodeRewardBreakdownReported && hasAppliedPidThisEpisode)
        {
            ReportEpisodeRewardBreakdown();
        }
    }

    private void ReportEpisodeRewardBreakdown()
    {
        if (episodeRewardBreakdownReported)
        {
            return;
        }

        StatsRecorder statsRecorder = Academy.Instance.StatsRecorder;
        statsRecorder.Add("PidTuningAgent/Reward Breakdown/Centered", episodeCenteredRewardTotal, StatAggregationMethod.Average);
        statsRecorder.Add("PidTuningAgent/Reward Breakdown/LowVelocity", episodeLowVelocityRewardTotal, StatAggregationMethod.Average);
        statsRecorder.Add("PidTuningAgent/Reward Breakdown/FailurePenalty", episodeFailurePenaltyTotal, StatAggregationMethod.Average);
        statsRecorder.Add(
            "PidTuningAgent/Reward Breakdown/TotalTracked",
            episodeCenteredRewardTotal +
            episodeLowVelocityRewardTotal +
            episodeFailurePenaltyTotal,
            StatAggregationMethod.Average);

        episodeRewardBreakdownReported = true;
    }

    private void ResolveReferences()
    {
        controller = controller == null ? FindFirstObjectByType<Controller>() : controller;

        if (controller != null)
        {
            pidControllerX = pidControllerX == null ? controller.pidControllerX : pidControllerX;
            pidControllerZ = pidControllerZ == null ? controller.pidControllerZ : pidControllerZ;
            ballSensor = ballSensor == null ? controller.ballSensor : ballSensor;
            plateTransform = plateTransform == null ? controller.plateTransform : plateTransform;
        }

        if (ballTransform == null && ballRigidbody != null)
        {
            ballTransform = ballRigidbody.transform;
        }

        if (ballRigidbody == null)
        {
            ballRigidbody = FindCandidateBallRigidbody();
            if (ballRigidbody != null)
            {
                ballTransform = ballRigidbody.transform;
            }
        }

        if (ballTransform == null && ballRigidbody != null)
        {
            ballTransform = ballRigidbody.transform;
        }
    }

    private Rigidbody FindCandidateBallRigidbody()
    {
        Rigidbody[] bodies = FindObjectsByType<Rigidbody>(FindObjectsSortMode.None);
        foreach (Rigidbody body in bodies)
        {
            if (plateTransform != null && body.transform == plateTransform)
            {
                continue;
            }

            return body;
        }

        return null;
    }

    private void ApplyBehaviorConfiguration()
    {
        behaviorParameters = behaviorParameters == null ? GetComponent<BehaviorParameters>() : behaviorParameters;
        decisionRequesterComponent = decisionRequesterComponent == null ? GetComponent<DecisionRequester>() : decisionRequesterComponent;

        if (behaviorParameters != null)
        {
            behaviorParameters.BehaviorName = behaviorName;
            behaviorParameters.BrainParameters.VectorObservationSize = ObservationSize;
            behaviorParameters.BrainParameters.NumStackedVectorObservations = 1;
            behaviorParameters.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(ContinuousActionSize);
        }

        if (decisionRequesterComponent != null)
        {
            decisionRequesterComponent.DecisionPeriod = Mathf.Max(1, decisionPeriod);
            decisionRequesterComponent.DecisionStep = 0;
            decisionRequesterComponent.TakeActionsBetweenDecisions = true;
            decisionRequesterComponent.enabled = !applyPidOnlyAtEpisodeStart;
        }
    }

    private Vector3 GetLocalBallPosition()
    {
        if (plateTransform == null || ballTransform == null)
        {
            return Vector3.zero;
        }

        return plateTransform.InverseTransformPoint(ballTransform.position);
    }

    private Vector3 GetLocalBallVelocity()
    {
        if (plateTransform == null || ballRigidbody == null)
        {
            return Vector3.zero;
        }

        return plateTransform.InverseTransformDirection(ballRigidbody.linearVelocity);
    }

    private static float NormalizeAngle(float angle)
    {
        if (angle > 180f)
        {
            angle -= 360f;
        }

        return angle;
    }

    private static float NormalizeGain(float value, Vector2 range)
    {
        if (Mathf.Approximately(range.x, range.y))
        {
            return 0f;
        }

        return Mathf.Lerp(-1f, 1f, Mathf.InverseLerp(range.x, range.y, value));
    }

    private static float NormalizeManualGain(float value, Vector2 range)
    {
        return Mathf.Clamp(NormalizeGain(value, range), -1f, 1f);
    }

    private static float ScaleAction(float actionValue, float minValue, float maxValue)
    {
        float normalized = Mathf.Clamp(actionValue, -1f, 1f);
        return Mathf.Lerp(minValue, maxValue, (normalized + 1f) * 0.5f);
    }

    private static float QuantizeGain(float value, Vector2 range, float step)
    {
        float clamped = Mathf.Clamp(value, range.x, range.y);
        if (step <= 0f)
        {
            return clamped;
        }

        float quantized = Mathf.Round((clamped - range.x) / step) * step + range.x;
        return Mathf.Clamp(quantized, range.x, range.y);
    }

    private void LogCurrentPidValues()
    {
        string pidX = pidControllerX == null
            ? "PID X: missing"
            : $"PID X => Kp={pidControllerX.Kp:F3}, Ki={pidControllerX.Ki:F3}, Kd={pidControllerX.Kd:F3}";

        string pidZ = pidControllerZ == null
            ? "PID Z: missing"
            : $"PID Z => Kp={pidControllerZ.Kp:F3}, Ki={pidControllerZ.Ki:F3}, Kd={pidControllerZ.Kd:F3}";

        Debug.Log($"[PidTuningAgent] Episode {episodeCounter} started with {pidX} | {pidZ}", this);
    }

    private void OnGUI()
    {
        if (!showPidOverlay)
        {
            return;
        }

        EnsureOverlayStyles();

        string pidX = pidControllerX == null
            ? "PID X: missing"
            : $"PID X  Kp {pidControllerX.Kp:F3}  Ki {pidControllerX.Ki:F3}  Kd {pidControllerX.Kd:F3}";

        string pidZ = pidControllerZ == null
            ? "PID Z: missing"
            : $"PID Z  Kp {pidControllerZ.Kp:F3}  Ki {pidControllerZ.Ki:F3}  Kd {pidControllerZ.Kd:F3}";
        string runtimeState =
            $"Radial {lastRadialDistance:F3}/{failDistance:F3}  " +
            $"Velocity {lastPlanarVelocity:F3}/{velocityReference:F3}  " +
            $"BallY {lastBallWorldY:F3}  FailY {lastFailWorldYThreshold:F3}  " +
            $"Reason {lastEpisodeEndReason}";

        Rect panelRect = new Rect(
            overlayPosition.x,
            overlayPosition.y,
            overlayPanelSize.x,
            overlayPanelSize.y);

        GUI.Box(panelRect, "PID Monitor", overlayBoxStyle);

        float lineHeight = overlayFontSize + 12f;
        GUI.Label(
            new Rect(panelRect.x + 12f, panelRect.y + 36f, panelRect.width - 24f, lineHeight),
            pidX,
            overlayLabelStyle);
        GUI.Label(
            new Rect(panelRect.x + 12f, panelRect.y + 36f + lineHeight, panelRect.width - 24f, lineHeight),
            pidZ,
            overlayLabelStyle);
        GUI.Label(
            new Rect(panelRect.x + 12f, panelRect.y + 36f + lineHeight * 2f, panelRect.width - 24f, lineHeight),
            runtimeState,
            overlayLabelStyle);
    }

    private void EnsureOverlayStyles()
    {
        int fontSize = Mathf.Max(overlayFontSize, 12);

        if (overlayBoxStyle == null)
        {
            overlayBoxStyle = new GUIStyle(GUI.skin.box);
            overlayBoxStyle.alignment = TextAnchor.UpperLeft;
            overlayBoxStyle.padding = new RectOffset(12, 12, 10, 10);
        }
        overlayBoxStyle.fontSize = fontSize;

        if (overlayLabelStyle == null)
        {
            overlayLabelStyle = new GUIStyle(GUI.skin.label);
            overlayLabelStyle.alignment = TextAnchor.MiddleLeft;
            overlayLabelStyle.normal.textColor = Color.white;
        }
        overlayLabelStyle.fontSize = fontSize;
    }

    private void OnDrawGizmosSelected()
    {
        if (!showResetGizmos)
        {
            return;
        }

        ResolveReferences();
        if (plateTransform == null)
        {
            return;
        }

        DrawResetGizmos();
    }

    private void DrawResetGizmos()
    {
        Vector3 plateCenter = plateTransform.position;
        Vector3 up = plateTransform.up;
        Vector3 right = plateTransform.right;
        Vector3 forward = plateTransform.forward;

        Vector3 spawnCenter = plateCenter + up * spawnHeightAbovePlate;
        Vector3 failFloorCenter = plateCenter - Vector3.up * failHeightBelowPlate;

        Gizmos.color = new Color(0.2f, 0.9f, 1f, 0.9f);
        DrawDisc(spawnCenter, right, forward, spawnRadius);

        Gizmos.color = new Color(1f, 0.45f, 0.1f, 0.9f);
        DrawDisc(plateCenter, right, forward, failDistance);

        Gizmos.color = new Color(0.2f, 0.9f, 1f, 0.35f);
        Gizmos.DrawLine(plateCenter, spawnCenter);

        Gizmos.color = new Color(1f, 0.2f, 0.2f, 0.35f);
        DrawDisc(failFloorCenter, Vector3.right, Vector3.forward, failDistance);
        Gizmos.DrawLine(plateCenter, failFloorCenter);
    }

    private static void DrawDisc(Vector3 center, Vector3 axisX, Vector3 axisY, float radius, int segments = 48)
    {
        if (radius <= 0f)
        {
            Gizmos.DrawSphere(center, 0.01f);
            return;
        }

        Vector3 previousPoint = center + axisX.normalized * radius;
        for (int i = 1; i <= segments; i++)
        {
            float angle = (i / (float)segments) * Mathf.PI * 2f;
            Vector3 nextPoint =
                center +
                axisX.normalized * Mathf.Cos(angle) * radius +
                axisY.normalized * Mathf.Sin(angle) * radius;
            Gizmos.DrawLine(previousPoint, nextPoint);
            previousPoint = nextPoint;
        }
    }
}
