using System;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace Unity.DemoTeam.Hair.Tests
{
	public class BoundaryBlendRotationTests
	{
		// Prior Strict/high differentials observed about 6e-7 rotation and 6.7e-6
		// history error. Allow rounding margin while bounding the final consumer
		// matrices, rather than requiring bit identity or only testing rotations.
		const float Accuracy = 2e-5f;
		bool previousEnabled;
		bool previousSynchronous;

		[SetUp]
		public void EnableSynchronousBurst()
		{
			previousEnabled = BurstCompiler.Options.EnableBurstCompilation;
			previousSynchronous = BurstCompiler.Options.EnableBurstCompileSynchronously;
			try
			{
				BurstCompiler.Options.EnableBurstCompilation = true;
				BurstCompiler.Options.EnableBurstCompileSynchronously = true;
				Assert.That(BurstCompiler.IsEnabled, Is.True, "These tests require Burst; do not silently test the managed fallback.");
			}
			catch
			{
				// NUnit does not call TearDown when SetUp fails.
				RestoreBurstOptions();
				throw;
			}
		}

		[TearDown]
		public void RestoreBurstOptions()
		{
			BurstCompiler.Options.EnableBurstCompileSynchronously = previousSynchronous;
			BurstCompiler.Options.EnableBurstCompilation = previousEnabled;
		}

		[TestCase(0, 0f)]
		[TestCase(0, 0.5f)]
		[TestCase(0, 1f)]
		[TestCase(1, 0f)]
		[TestCase(1, 0.25f)]
		[TestCase(1, 0.5f)]
		[TestCase(1, 1f)]
		[TestCase(3, 0.5f)]
		public void RandomNonuniformTRSMatchesManagedFormula(int stepCount, float frameFracHi)
		{
			const int count = 1024;
			var data = new Inputs(count, 41791);
			var expectedRotations = new Quaternion[count];
			var expectedHistory = (Matrix4x4[])data.history.Clone();
			ManagedReference(data.relative, data.remap, data.current, expectedRotations, expectedHistory, count, stepCount > 0, 1f - frameFracHi);
			using (var native = new NativeInputs(data))
			{
				BindRunner(native, count, stepCount, frameFracHi)();
				Compare(native, expectedRotations, expectedHistory, count, stepCount == 0, "random", stepCount, frameFracHi);
			}
		}

		[TestCase(0)]
		[TestCase(8)]
		[TestCase(32)]
		[TestCase(103)]
		[TestCase(256)]
		public void RemappedHistoryTransitionsDoNotAdvanceOnNoStep(int count)
		{
			const int capacity = 256;
			var data = new Inputs(capacity, 92417);
			var expectedHistory = (Matrix4x4[])data.history.Clone();
			var expectedRotations = new Quaternion[capacity];
			var random = new System.Random(51239);
			var steps = new[] { 1, 0, 2, 0, 1, 0 };
			var fractions = new[] { 0f, 0.25f, 0.5f, 1f, 1f, 0.5f };
			using (var native = new NativeInputs(data))
			{
				for (int frame = 0; frame != steps.Length; frame++)
				{
					var before = native.history.ToArray();
					for (int i = 0; i != capacity; i++)
					{
						data.current[i] = RandomTRS(random);
						data.remap[i] = (i + frame) % 17 == 0 ? -1 : (capacity - 1 - i + frame) % capacity;
						var j = data.remap[i];
						data.relative[i] = j < 0 ? Matrix4x4.identity : Matrix4x4.Inverse(data.current[i]) * expectedHistory[j];
						native.relative[i] = j < 0 ? Matrix4x4.identity : Matrix4x4.Inverse(data.current[i]) * before[j];
					}
					native.current.CopyFrom(data.current);
					native.remap.CopyFrom(data.remap);
					ManagedReference(data.relative, data.remap, data.current, expectedRotations, expectedHistory, count, steps[frame] > 0, 1f - fractions[frame]);
					BindRunner(native, count, steps[frame], fractions[frame])();
					Compare(native, expectedRotations, expectedHistory, count, false, "transition-" + frame, steps[frame], fractions[frame]);
					if (steps[frame] == 0)
					{
						for (int i = 0; i != capacity; i++)
							Assert.That(native.history[i].Equals(before[i]), Is.True, "History advanced without a step at " + i);
					}
					for (int i = count; i != capacity; i++)
						Assert.That(native.history[i].Equals(before[i]), Is.True, "Inactive history was overwritten at " + i);
				}
			}
		}

		[Test, Explicit("Paired microbenchmark: run explicitly with Burst enabled in an otherwise idle editor.")]
		public void BenchmarkManagedAndNativeBoundaryInterpolation()
		{
			const int warmup = 16;
			const int pairs = 12;
			const int iterations = 128;
			const int allocationIterations = 128;
			foreach (var count in new[] { 8, 32, 103, 256 })
			{
				var data = new Inputs(count, 41791 + count);
				var rotations = new Quaternion[count];
				var history = (Matrix4x4[])data.history.Clone();
				ManagedReference(data.relative, data.remap, data.current, rotations, history, count, true, 0.5f);
				using (var native = new NativeInputs(data, Allocator.Persistent))
				{
					Action managed = () => ManagedPointerWork(native, count);
					// Reflection and closure binding are outside all timed regions.
					// Both production Run calls are included, not Execute called as C#.
					var run = BindRunner(native, count, 1, 0.5f);
					for (int i = 0; i != warmup; i++)
					{
						managed();
						run();
						AllocateFrameBuffers(Allocator.Temp);
						AllocateFrameBuffers(Allocator.TempJob);
					}
					for (int pair = 0; pair != pairs; pair++)
					{
						double managedMs, nativeMs;
						if ((pair & 1) == 0)
						{
							managedMs = Measure(managed, iterations);
							nativeMs = Measure(run, iterations);
						}
						else
						{
							nativeMs = Measure(run, iterations);
							managedMs = Measure(managed, iterations);
						}
						// Validate Burst output even when the baseline ran last.
						run();
						Compare(native, rotations, history, count, false, "benchmark-" + pair, 1, 0.5f);
						// Match the six fixed-capacity frame buffers in production, including
						// clear/dispose. Only the four job inputs/outputs change allocator.
						// Persistent history, data copies, and job work are excluded.
						double baselineAllocationMs, burstAllocationMs;
						if ((pair & 1) == 0)
						{
							baselineAllocationMs = Measure(() => AllocateFrameBuffers(Allocator.Temp), allocationIterations);
							burstAllocationMs = Measure(() => AllocateFrameBuffers(Allocator.TempJob), allocationIterations);
						}
						else
						{
							burstAllocationMs = Measure(() => AllocateFrameBuffers(Allocator.TempJob), allocationIterations);
							baselineAllocationMs = Measure(() => AllocateFrameBuffers(Allocator.Temp), allocationIterations);
						}
						TestContext.WriteLine(string.Format(CultureInfo.InvariantCulture,
							"BOUNDARY_BENCH contacts={0} pair={1} warmup={2} iterations={3} first={4} managed_pointer_work_ms={5:R} native_run_ms={6:R} baseline_six_buffer_alloc_dispose_ms={7:R} burst_six_buffer_alloc_dispose_ms={8:R} burst={9} alloc_capacity={10} alloc_buffers=6 history_allocator=Persistent synthetic_workload={11} allocation_iterations={12} work_allocator=Persistent",
							count, pair, warmup, iterations, (pair & 1) == 0 ? "managed" : "native", managedMs, nativeMs, baselineAllocationMs, burstAllocationMs, BurstCompiler.IsEnabled, HairSim.Conf.MAX_BOUNDARIES, count > HairSim.Conf.MAX_BOUNDARIES, allocationIterations));
					}
				}
			}
		}

		// Original managed SVD and history loops over the same native buffers used
		// by the jobs. Matrix inverse/product preparation is outside both timings.
		static unsafe void ManagedPointerWork(NativeInputs data, int count)
		{
			var ptrRemap = (int*)data.remap.GetUnsafePtr();
			var ptrMatrix = (Matrix4x4*)data.current.GetUnsafePtr();
			var ptrMatrixA = (Matrix4x4*)data.relative.GetUnsafePtr();
			var ptrMatrixQ = (Quaternion*)data.rotations.GetUnsafePtr();
			var ptrMatrixPrev = (Matrix4x4*)data.history.GetUnsafePtr();
			for (int i = 0; i != count; i++)
			{
				ptrMatrixQ[i] = ptrRemap[i] != -1
					? (Quaternion)svd.svdRotation((float3x3)((float4x4)ptrMatrixA[i]))
					: Quaternion.identity;
			}
			for (int i = 0; i != count; i++)
			{
				ref var Ma = ref ptrMatrix[i];
				ref var M = ref ptrMatrixA[i];
				ref var q = ref ptrMatrixQ[i];
				var M_t = AffineUtility.AffineInterpolate4x4(M, ((quaternion)q).value, 0.5f);
				ptrMatrixPrev[i] = AffineUtility.AffineMul4x4(Ma, M_t);
			}
		}

		// Independent managed execution of the pre-job formula. Inverse/product
		// preparation is deliberately excluded from both benchmark work regions.
		static void ManagedReference(Matrix4x4[] relative, int[] remap, Matrix4x4[] current, Quaternion[] rotations, Matrix4x4[] history, int count, bool update, float fraction)
		{
			for (int i = 0; i != count; i++)
				rotations[i] = remap[i] < 0 ? Quaternion.identity : (Quaternion)svd.svdRotation((float3x3)(float4x4)relative[i]);
			if (update)
			{
				for (int i = 0; i != count; i++)
				{
					var interpolated = AffineUtility.AffineInterpolate4x4(relative[i], ((quaternion)rotations[i]).value, fraction);
					history[i] = AffineUtility.AffineMul4x4(current[i], interpolated);
				}
			}
		}

		delegate void UpdateHistory(NativeArray<Matrix4x4> matrices, NativeArray<Quaternion> rotations, NativeArray<Matrix4x4> currentMatrices, NativeArray<Matrix4x4> previousMatrices, int count, int stepCount, float frameFracHi);

		static Action BindRunner(NativeInputs data, int count, int stepCount, float frameFracHi)
		{
			var type = typeof(HairSim).GetNestedType("BoundaryBlendRotationJob", BindingFlags.NonPublic);
			Assert.That(type, Is.Not.Null, "Production boundary job not found.");
			var boxed = Activator.CreateInstance(type);
			type.GetField("matrices").SetValue(boxed, data.relative);
			type.GetField("remap").SetValue(boxed, data.remap);
			type.GetField("rotations").SetValue(boxed, data.rotations);
			var rotate = (Action)typeof(BoundaryBlendRotationTests).GetMethod(nameof(CaptureRunner), BindingFlags.NonPublic | BindingFlags.Static)
				.MakeGenericMethod(type).Invoke(null, new object[] { boxed, count });
			var historyMethod = typeof(HairSim).GetMethod("UpdateBoundaryHistory", BindingFlags.NonPublic | BindingFlags.Static);
			Assert.That(historyMethod, Is.Not.Null, "Production history update not found.");
			var updateHistory = (UpdateHistory)Delegate.CreateDelegate(typeof(UpdateHistory), historyMethod);
			return () =>
			{
				rotate();
				// Exercise the production step-count guard and frameFracHi conversion,
				// including the no-step path, rather than reproducing them in the test.
				updateHistory(data.relative, data.rotations, data.current, data.history, count, stepCount, frameFracHi);
			};
		}

		static Action CaptureRunner<T>(object boxed, int count) where T : struct, IJobParallelFor
		{
			var job = (T)boxed;
			return () => job.Run(count);
		}

		static void Compare(NativeInputs native, Quaternion[] rotations, Matrix4x4[] history, int count, bool unchanged, string label, int steps, float frameFracHi)
		{
			float rotationError = 0f, historyError = 0f;
			int missing = 0;
			for (int i = 0; i != count; i++)
			{
				if (native.remap[i] < 0)
				{
					missing++;
					Assert.That(native.rotations[i].Equals(Quaternion.identity), Is.True, "Missing remap must use identity rotation at " + i);
				}
				var a = ((quaternion)native.rotations[i]).value;
				var b = ((quaternion)rotations[i]).value;
				var error = math.min(math.length(a - b), math.length(a + b));
				Assert.That(float.IsNaN(error) || float.IsInfinity(error), Is.False, "Nonfinite rotation at " + i);
				rotationError = Mathf.Max(rotationError, error);
				if (unchanged)
					Assert.That(native.history[i].Equals(history[i]), Is.True, "History advanced without a step at " + i);
				for (int element = 0; element != 16; element++)
				{
					var delta = Mathf.Abs(native.history[i][element] - history[i][element]);
					Assert.That(float.IsNaN(delta) || float.IsInfinity(delta), Is.False, "Nonfinite history at " + i);
					historyError = Mathf.Max(historyError, delta);
				}
			}
			TestContext.WriteLine(string.Format(CultureInfo.InvariantCulture,
				"BOUNDARY_DIFF case={0} contacts={1} steps={2} frame_frac_hi={3:R} max_rotation_error={4:R} max_history_element_error={5:R} tolerance={6:R} burst={7} mapped={8} missing={9}",
				label, count, steps, frameFracHi, rotationError, historyError, Accuracy, BurstCompiler.IsEnabled, count - missing, missing));
			Assert.That(rotationError, Is.LessThanOrEqualTo(Accuracy));
			Assert.That(historyError, Is.LessThanOrEqualTo(Accuracy));
		}

		static double Measure(Action action, int iterations)
		{
			var clock = Stopwatch.StartNew();
			for (int i = 0; i != iterations; i++) action();
			clock.Stop();
			return clock.Elapsed.TotalMilliseconds;
		}

		static void AllocateFrameBuffers(Allocator workAllocator)
		{
			const int capacity = HairSim.Conf.MAX_BOUNDARIES;
			using (var remap = new NativeArray<int>(capacity, workAllocator, NativeArrayOptions.ClearMemory))
			using (var handle = new NativeArray<int>(capacity, Allocator.Temp, NativeArrayOptions.ClearMemory))
			using (var current = new NativeArray<Matrix4x4>(capacity, workAllocator, NativeArrayOptions.ClearMemory))
			using (var relative = new NativeArray<Matrix4x4>(capacity, workAllocator, NativeArrayOptions.ClearMemory))
			using (var rotations = new NativeArray<Quaternion>(capacity, workAllocator, NativeArrayOptions.ClearMemory))
			using (var shape = new NativeArray<HairBoundary.RuntimeShape.Data>(capacity, Allocator.Temp, NativeArrayOptions.ClearMemory)) { }
		}

		static float Next(System.Random random, float min, float max) => min + (max - min) * (float)random.NextDouble();
		static Vector3 Point(System.Random random, float scale) => new Vector3(Next(random, -scale, scale), Next(random, -scale, scale), Next(random, -scale, scale));
		static Matrix4x4 RandomTRS(System.Random random) => Matrix4x4.TRS(Point(random, 2f), Quaternion.Euler(Point(random, 180f)), new Vector3(Next(random, 0.5f, 2f), Next(random, 0.5f, 2f), Next(random, 0.5f, 2f)));

		sealed class Inputs
		{
			public readonly Matrix4x4[] relative, current, history;
			public readonly int[] remap;

			public Inputs(int count, int seed)
			{
				relative = new Matrix4x4[count];
				current = new Matrix4x4[count];
				history = new Matrix4x4[count];
				remap = new int[count];
				var random = new System.Random(seed);
				for (int i = 0; i != count; i++)
				{
					current[i] = RandomTRS(random);
					history[i] = RandomTRS(random);
					remap[i] = i % 17 == 0 ? -1 : count - 1 - i;
				}
				for (int i = 0; i != count; i++)
					relative[i] = remap[i] < 0 ? Matrix4x4.identity : Matrix4x4.Inverse(current[i]) * history[remap[i]];
			}
		}

		sealed class NativeInputs : IDisposable
		{
			public NativeArray<Matrix4x4> relative, current, history;
			public NativeArray<int> remap;
			public NativeArray<Quaternion> rotations;

			public NativeInputs(Inputs data, Allocator workAllocator = Allocator.TempJob)
			{
				relative = new NativeArray<Matrix4x4>(data.relative, workAllocator);
				current = new NativeArray<Matrix4x4>(data.current, workAllocator);
				history = new NativeArray<Matrix4x4>(data.history, Allocator.Persistent);
				remap = new NativeArray<int>(data.remap, workAllocator);
				rotations = new NativeArray<Quaternion>(data.remap.Length, workAllocator);
			}

			public void Dispose()
			{
				rotations.Dispose();
				remap.Dispose();
				history.Dispose();
				current.Dispose();
				relative.Dispose();
			}
		}
	}
}
