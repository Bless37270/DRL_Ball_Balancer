using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;

[RequireComponent(typeof(BehaviorParameters))]
[RequireComponent(typeof(DecisionRequester))]
public class DDPGFC350EPidAgent : Agent
{
    [Header("Control Graph")]
    [SerializeField] private Controller controller;
    [SerializeField] private PIDController pidControllerX;
    [SerializeField] private PIDController pidControllerZ;
    [SerializeField] private Transform plateTransform;

    [Header("Optional Reset References")]
    [SerializeField] private Rigidbody ballRigidbody;
    [SerializeField] private Transform ballTransform;

    [Header("Behavior")]
    [SerializeField] private string behaviorName = "ddpg_fc_350_e_pid_agent";
    [SerializeField] private int decisionPeriod = 5;
    [SerializeField] private int maxEpisodeSteps = 3000;
    [SerializeField] private bool applyPidOnlyAtEpisodeStart = true;

    [Header("PID Gain Ranges")]
    [SerializeField] private Vector2 kpRange = new Vector2(0f, 10f);
    [SerializeField] private Vector2 kiRange = new Vector2(0f, 2f);
    [SerializeField] private Vector2 kdRange = new Vector2(0f, 6f);
    [SerializeField] private float minimumActiveKp = 1.5f;

    [Header("PID Quantization")]
    [SerializeField] private float kpStep = 0.0001f;
    [SerializeField] private float kiStep = 0.0001f;
    [SerializeField] private float kdStep = 0.0001f;

    [Header("Reference PID Seed")]
    [SerializeField] private bool seedWithPaperEPidGains = true;
    [SerializeField] private Vector3 paperEPidGains = new Vector3(4.2025f, 1.2529f, 5.1323f);
    [SerializeField] private bool useSeedWhenInferenceOutputsMinimum = true;

    [Header("Reset")]
    [SerializeField] private float spawnRadius = 0.25f;
    [SerializeField] private float minimumSpawnRadius = 0.05f;
    [SerializeField] private float spawnHeightAbovePlate = 0.1f;
    [SerializeField] private float failDistance = 0.55f;
    [SerializeField] private float failHeightBelowPlate = 0.3f;

    [Header("Rewards")]
    [SerializeField] private float centeredRewardWeight = 0.015f;
    [SerializeField] private float lowVelocityRewardWeight = 0.003f;
    [SerializeField] private float pidEffortPenaltyWeight = 0.0002f;
    [SerializeField] private float distanceRewardReference = 0f;
    [SerializeField] private float observationDistanceReference = 0f;
    [SerializeField] private float centerProgressRewardWeight = 0.02f;
    [SerializeField] private float outwardVelocityPenaltyWeight = 0.01f;
    [SerializeField] private float velocityReference = 5f;
    [SerializeField] private float failurePenalty = -10f;

    private const int ObservationSize = 10;
    private const int ContinuousActionSize = 3;

    private BehaviorParameters behaviorParameters;
    private DecisionRequester decisionRequester;

    private Vector3 initialPlatePosition;
    private Quaternion initialPlateRotation;
    private Quaternion initialBallRotation;

    private Vector2 integratedError;
    private Vector2 previousError;
    private Vector2 currentError;
    private Vector2 currentDerivative;
    private float previousRewardRadialDistance;
    private bool hasPreviousRewardDistance;
    private bool hasPreviousError;
    private bool hasAppliedPidThisEpisode;
    private bool pendingEpisodeDecision;
    private float episodeCenteredRewardTotal;
    private float episodeLowVelocityRewardTotal;
    private float episodeFailurePenaltyTotal;
    private float episodePidEffortPenaltyTotal;
    private bool episodeRewardBreakdownReported = true;

    public override void Initialize()
    {
        ResolveReferences();

        behaviorParameters = GetComponent<BehaviorParameters>();
        decisionRequester = GetComponent<DecisionRequester>();

        ApplyBehaviorConfiguration();

        if (plateTransform != null)
        {
            initialPlatePosition = plateTransform.position;
            initialPlateRotation = plateTransform.rotation;
        }

        if (ballTransform != null)
        {
            initialBallRotation = ballTransform.rotation;
        }

        MaxStep = maxEpisodeSteps;
        ResetPidState();
        ApplyPaperSeedIfRequested();
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

        if (controller != null)
        {
            controller.targetPosition = Vector2.zero;
        }

        ResetErrorState();
        ResetRewardDistanceState();
        ResetPidState();
        ApplyPaperSeedIfRequested();
        ResetPlate();
        ResetBall();

        hasAppliedPidThisEpisode = false;
        pendingEpisodeDecision = true;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        ResolveReferences();
        EPidState ePidState = new EPidState(currentError, integratedError, currentDerivative);
        float observationDistanceScale = GetObservationDistanceScale();

        sensor.AddObservation(Mathf.Clamp(ePidState.error.x / observationDistanceScale, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(ePidState.error.y / observationDistanceScale, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(ePidState.integral.x / observationDistanceScale, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(ePidState.integral.y / observationDistanceScale, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(ePidState.derivative.x / velocityReference, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(ePidState.derivative.y / velocityReference, -1f, 1f));
        sensor.AddObservation(NormalizeGain(pidControllerX != null ? pidControllerX.Kp : paperEPidGains.x, kpRange));
        sensor.AddObservation(NormalizeGain(pidControllerX != null ? pidControllerX.Ki : paperEPidGains.y, kiRange));
        sensor.AddObservation(NormalizeGain(pidControllerX != null ? pidControllerX.Kd : paperEPidGains.z, kdRange));
        sensor.AddObservation(Mathf.Clamp01(ePidState.error.magnitude / observationDistanceScale));
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
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> continuous = actionsOut.ContinuousActions;
        if (continuous.Length < ContinuousActionSize)
        {
            return;
        }

        Vector3 gains = seedWithPaperEPidGains ? paperEPidGains : Vector3.zero;
        if (pidControllerX != null)
        {
            gains = new Vector3(pidControllerX.Kp, pidControllerX.Ki, pidControllerX.Kd);
        }

        continuous[0] = NormalizeManualGain(gains.x, kpRange);
        continuous[1] = NormalizeManualGain(gains.y, kiRange);
        continuous[2] = NormalizeManualGain(gains.z, kdRange);
    }

    private void FixedUpdate()
    {
        UpdateErrorState();

        if (applyPidOnlyAtEpisodeStart && pendingEpisodeDecision)
        {
            RequestDecision();
            pendingEpisodeDecision = false;
        }

        if (!hasAppliedPidThisEpisode)
        {
            return;
        }

        float radialDistance = currentError.magnitude;
        float planarVelocity = GetPlanarVelocityMagnitude();
        float rewardDistanceScale = GetRewardDistanceScale();
        float normalizedDistance = Mathf.Clamp01(radialDistance / rewardDistanceScale);
        float normalizedVelocity = Mathf.Clamp01(planarVelocity / Mathf.Max(velocityReference, 0.0001f));

        float pidEffort = GetNormalizedPidEffort();
        float centeredStepReward = (1f - normalizedDistance * normalizedDistance) * centeredRewardWeight;
        float lowVelocityStepReward = (1f - normalizedVelocity) * lowVelocityRewardWeight;
        float centerProgressReward = GetCenterProgressReward(radialDistance, rewardDistanceScale);
        float outwardVelocityPenalty = GetOutwardVelocityPenalty(rewardDistanceScale);
        float pidEffortPenalty = pidEffort * pidEffortPenaltyWeight;
        float stepReward =
            centeredStepReward +
            lowVelocityStepReward -
            outwardVelocityPenalty +
            centerProgressReward -
            pidEffortPenalty;

        AddTrackedReward(ref episodeCenteredRewardTotal, centeredStepReward);
        AddTrackedReward(ref episodeLowVelocityRewardTotal, lowVelocityStepReward);
        AddTrackedReward(ref episodePidEffortPenaltyTotal, -pidEffortPenalty);

        if (HasFailed(radialDistance))
        {
            AddTrackedReward(ref episodeFailurePenaltyTotal, failurePenalty);
            AddReward(failurePenalty);
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
        episodePidEffortPenaltyTotal = 0f;
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

        float totalTracked =
            episodeCenteredRewardTotal +
            episodeLowVelocityRewardTotal +
            episodeFailurePenaltyTotal +
            episodePidEffortPenaltyTotal;

        StatsRecorder statsRecorder = Academy.Instance.StatsRecorder;
        statsRecorder.Add("DDPGFC350EPidAgent/Reward Breakdown/Centered", episodeCenteredRewardTotal, StatAggregationMethod.Average);
        statsRecorder.Add("DDPGFC350EPidAgent/Reward Breakdown/LowVelocity", episodeLowVelocityRewardTotal, StatAggregationMethod.Average);
        statsRecorder.Add("DDPGFC350EPidAgent/Reward Breakdown/FailurePenalty", episodeFailurePenaltyTotal, StatAggregationMethod.Average);
        statsRecorder.Add("DDPGFC350EPidAgent/Reward Breakdown/TotalTracked", totalTracked, StatAggregationMethod.Average);

        episodeRewardBreakdownReported = true;
    }

    private void ApplyPidActions(ActionSegment<float> continuousActions)
    {
        if (pidControllerX == null || pidControllerZ == null)
        {
            return;
        }

        float kpMin = Mathf.Clamp(minimumActiveKp, kpRange.x, kpRange.y);
        float sharedKp = QuantizeGain(ScaleAction(continuousActions[0], kpMin, kpRange.y), kpRange, kpStep);
        float sharedKi = QuantizeGain(ScaleAction(continuousActions[1], kiRange.x, kiRange.y), kiRange, kiStep);
        float sharedKd = QuantizeGain(ScaleAction(continuousActions[2], kdRange.x, kdRange.y), kdRange, kdStep);

        if (ShouldUseSeedFallback(continuousActions))
        {
            sharedKp = QuantizeGain(paperEPidGains.x, kpRange, kpStep);
            sharedKi = QuantizeGain(paperEPidGains.y, kiRange, kiStep);
            sharedKd = QuantizeGain(paperEPidGains.z, kdRange, kdStep);
        }

        pidControllerX.Kp = sharedKp;
        pidControllerX.Ki = sharedKi;
        pidControllerX.Kd = sharedKd;

        pidControllerZ.Kp = sharedKp;
        pidControllerZ.Ki = sharedKi;
        pidControllerZ.Kd = sharedKd;
    }

    private bool ShouldUseSeedFallback(ActionSegment<float> continuousActions)
    {
        if (!useSeedWhenInferenceOutputsMinimum || Academy.Instance.IsCommunicatorOn)
        {
            return false;
        }

        return continuousActions[0] <= -0.999f &&
            continuousActions[1] <= -0.999f &&
            continuousActions[2] <= -0.999f;
    }

    private void UpdateErrorState()
    {
        Vector3 localBallPosition = GetLocalBallPosition();
        Vector2 error = new Vector2(-localBallPosition.x, -localBallPosition.z);

        float dt = Mathf.Max(Time.fixedDeltaTime, 0.0001f);
        integratedError += error * dt;
        integratedError.x = Mathf.Clamp(integratedError.x, -failDistance, failDistance);
        integratedError.y = Mathf.Clamp(integratedError.y, -failDistance, failDistance);

        currentDerivative = hasPreviousError ? (error - previousError) / dt : Vector2.zero;
        previousError = error;
        currentError = error;
        hasPreviousError = true;
    }

    private void ResolveReferences()
    {
        controller = controller == null ? FindFirstObjectByType<Controller>() : controller;

        if (controller != null)
        {
            pidControllerX = pidControllerX == null ? controller.pidControllerX : pidControllerX;
            pidControllerZ = pidControllerZ == null ? controller.pidControllerZ : pidControllerZ;
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

        if (controller != null && controller.ballTransform == null && ballTransform != null)
        {
            controller.ballTransform = ballTransform;
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
        decisionRequester = decisionRequester == null ? GetComponent<DecisionRequester>() : decisionRequester;

        if (behaviorParameters != null)
        {
            behaviorParameters.BehaviorName = behaviorName;
            behaviorParameters.BrainParameters.VectorObservationSize = ObservationSize;
            behaviorParameters.BrainParameters.NumStackedVectorObservations = 1;
            behaviorParameters.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(ContinuousActionSize);
        }

        if (decisionRequester != null)
        {
            decisionRequester.DecisionPeriod = Mathf.Max(1, decisionPeriod);
            decisionRequester.DecisionStep = 0;
            decisionRequester.TakeActionsBetweenDecisions = true;
            decisionRequester.enabled = !applyPidOnlyAtEpisodeStart;
        }
    }

    private bool HasFailed(float radialDistance)
    {
        if (radialDistance > failDistance)
        {
            return true;
        }

        if (ballTransform == null || plateTransform == null)
        {
            return false;
        }

        return ballTransform.position.y < plateTransform.position.y - failHeightBelowPlate;
    }

    private void ResetBall()
    {
        if (plateTransform == null || ballTransform == null)
        {
            return;
        }

        Vector2 offset = GetSpawnOffset();
        Vector3 spawnPosition = plateTransform.TransformPoint(new Vector3(offset.x, spawnHeightAbovePlate, offset.y));
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
            return Vector2.right * Mathf.Max(innerRadius, 0.05f);
        }

        float angle = Random.Range(0f, Mathf.PI * 2f);
        float radius = Mathf.Sqrt(Random.Range(innerRadius * innerRadius, outerRadius * outerRadius));
        return new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)) * radius;
    }

    private void ResetPlate()
    {
        if (plateTransform != null)
        {
            plateTransform.SetPositionAndRotation(initialPlatePosition, initialPlateRotation);
        }
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

    private void ApplyPaperSeedIfRequested()
    {
        if (!seedWithPaperEPidGains || pidControllerX == null || pidControllerZ == null)
        {
            return;
        }

        pidControllerX.Kp = paperEPidGains.x;
        pidControllerX.Ki = paperEPidGains.y;
        pidControllerX.Kd = paperEPidGains.z;

        pidControllerZ.Kp = paperEPidGains.x;
        pidControllerZ.Ki = paperEPidGains.y;
        pidControllerZ.Kd = paperEPidGains.z;
    }

    private void ResetErrorState()
    {
        integratedError = Vector2.zero;
        previousError = Vector2.zero;
        currentError = Vector2.zero;
        currentDerivative = Vector2.zero;
        hasPreviousError = false;
    }

    private void ResetRewardDistanceState()
    {
        previousRewardRadialDistance = 0f;
        hasPreviousRewardDistance = false;
    }

    private Vector3 GetLocalBallPosition()
    {
        if (plateTransform == null || ballTransform == null)
        {
            return Vector3.zero;
        }

        return plateTransform.InverseTransformPoint(ballTransform.position);
    }

    private float GetPlanarVelocityMagnitude()
    {
        if (plateTransform == null || ballRigidbody == null)
        {
            return 0f;
        }

        Vector3 localVelocity = plateTransform.InverseTransformDirection(ballRigidbody.linearVelocity);
        return new Vector2(localVelocity.x, localVelocity.z).magnitude;
    }

    private float GetNormalizedPidEffort()
    {
        if (pidControllerX == null)
        {
            return 0f;
        }

        float kp = Mathf.InverseLerp(kpRange.x, kpRange.y, pidControllerX.Kp);
        float ki = Mathf.InverseLerp(kiRange.x, kiRange.y, pidControllerX.Ki);
        float kd = Mathf.InverseLerp(kdRange.x, kdRange.y, pidControllerX.Kd);
        return (kp + ki + kd) / 3f;
    }

    private float GetCenterProgressReward(float radialDistance, float rewardDistanceScale)
    {
        if (!hasPreviousRewardDistance)
        {
            previousRewardRadialDistance = radialDistance;
            hasPreviousRewardDistance = true;
            return 0f;
        }

        float progressTowardCenter = (previousRewardRadialDistance - radialDistance) / Mathf.Max(rewardDistanceScale, 0.0001f);
        previousRewardRadialDistance = radialDistance;
        return Mathf.Clamp(progressTowardCenter, -1f, 1f) * centerProgressRewardWeight;
    }

    private float GetOutwardVelocityPenalty(float rewardDistanceScale)
    {
        if (currentError.sqrMagnitude <= 0.000001f)
        {
            return 0f;
        }

        float outwardVelocity = Vector2.Dot(currentError.normalized, currentDerivative);
        float normalizedOutwardVelocity = Mathf.Clamp01(outwardVelocity / Mathf.Max(velocityReference, rewardDistanceScale, 0.0001f));
        return normalizedOutwardVelocity * outwardVelocityPenaltyWeight;
    }

    private float GetRewardDistanceScale()
    {
        if (distanceRewardReference > 0f)
        {
            return Mathf.Max(distanceRewardReference, 0.0001f);
        }

        float spawnScale = Mathf.Max(spawnRadius, minimumSpawnRadius, 0.0001f);
        return Mathf.Min(Mathf.Max(failDistance, 0.0001f), spawnScale);
    }

    private float GetObservationDistanceScale()
    {
        if (observationDistanceReference > 0f)
        {
            return Mathf.Max(observationDistanceReference, 0.0001f);
        }

        return GetRewardDistanceScale();
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

    private readonly struct EPidState
    {
        public EPidState(Vector2 error, Vector2 integral, Vector2 derivative)
        {
            this.error = error;
            this.integral = integral;
            this.derivative = derivative;
        }

        public readonly Vector2 error;
        public readonly Vector2 integral;
        public readonly Vector2 derivative;
    }
}
