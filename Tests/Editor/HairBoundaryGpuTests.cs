using System.Collections.Generic;
using System.Runtime.InteropServices;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace Unity.DemoTeam.Hair.Tests
{
	// Run in an Editor with an actual compute-capable graphics device, not -nographics.
	// The oracle is independent exhaustive GPU traversal of the existing primitives.
	// Coincident capsule endpoints are outside SdCapsule's defined domain in this
	// change: every generated capsule has a nonzero segment, including zero radii.
	public class HairBoundaryGpuTests
	{
		const int Capacity = 256;
		const int ResultsPerPoint = 8;
		const float Epsilon = 1e-4f;
		const string PackageRoot = "Packages/com.unity.demoteam.hair/";

		[StructLayout(LayoutKind.Sequential)]
		struct Shape
		{
			public Vector3 pA;
			public float radius;
			public Vector3 pB;
			public float extra;
		}

		enum Variation { Ordinary, Mixed, Ties, NegativeRadius, Extreme, Transformed }

		sealed class Scenario
		{
			public string name;
			public Shape[] shapes;
			public Vector3[] points;
			public HairSim.VolumeCBufferEnvironment environment;
			public int capsuleCount;
			public Variation variation;
		}

		ComputeShader m_volume;
		ComputeShader m_queries;
		int m_validateKernel;
		ComputeBuffer m_shapes;
		ComputeBuffer m_matricesInv;
		ComputeBuffer m_environment;
		Texture3D m_sdf;
		readonly int m_environmentID = Shader.PropertyToID("VolumeCBufferEnvironment");
		readonly int m_environmentBytes = Marshal.SizeOf(typeof(HairSim.VolumeCBufferEnvironment));

		[SetUp]
		public void SetUp()
		{
			if (!SystemInfo.supportsComputeShaders || SystemInfo.graphicsDeviceType == GraphicsDeviceType.Null)
				Assert.Ignore("Requires a real compute-capable GPU; do not use -nographics.");
			if (!SystemInfo.SupportsTextureFormat(TextureFormat.RFloat))
				Assert.Ignore("Requires RFloat Texture3D support for the mixed discrete boundary oracle.");

			m_volume = AssetDatabase.LoadAssetAtPath<ComputeShader>(PackageRoot + "Runtime/HairSimComputeVolume.compute");
			m_queries = AssetDatabase.LoadAssetAtPath<ComputeShader>(PackageRoot + "Tests/Editor/HairBoundaryGpuTests.compute");
			Assert.That(m_volume, Is.Not.Null, "Production volume compute asset must be imported.");
			Assert.That(m_queries, Is.Not.Null, "GPU oracle compute asset must be imported.");
			m_validateKernel = m_queries.FindKernel("Validate");
			m_shapes = new ComputeBuffer(Capacity, 32);
			m_matricesInv = new ComputeBuffer(Capacity, 64);
			m_environment = new ComputeBuffer(1, m_environmentBytes, ComputeBufferType.Constant);
			m_matricesInv.SetData(IdentityMatrices());
			m_sdf = new Texture3D(2, 2, 2, TextureFormat.RFloat, false)
			{
				filterMode = FilterMode.Bilinear,
				wrapMode = TextureWrapMode.Clamp,
			};
			var pixels = new Color[8];
			for (int i = 0; i < pixels.Length; i++) pixels[i] = new Color(0.25f, 0, 0, 0);
			m_sdf.SetPixels(pixels);
			m_sdf.Apply();
		}

		[TearDown]
		public void TearDown()
		{
			m_environment?.Release();
			m_matricesInv?.Release();
			m_shapes?.Release();
			if (m_sdf != null) UnityEngine.Object.DestroyImmediate(m_sdf);
		}

		[Test]
		[Category("HairBoundaryGpuCorrectness")]
		public void FlatFusedQueriesMatchExhaustiveGpuTraversal()
		{
			// Reuse the same buffers through small, large, empty and mixed sets.
			// Count 32 is added to the original 15-case corpus; extra random points in
			// the transformed case keep its total at 69,030 query locations.
			int[] counts = { 0, 1, 8, 9, 32, 103, 256, 103, 9, 0, 103, 103, 103, 103, 103 };
			int total = 0;
			for (int i = 0; i < counts.Length; i++)
			{
				var variation = i < 10 ? Variation.Ordinary : (Variation)(i - 9);
				var scenario = MakeScenario(counts[i], variation, 73103 + i, i == 14 ? 1344 : 0);
				Upload(scenario);
				using (var commands = new CommandBuffer())
				{
					Validate(commands, scenario);
				}
				total += scenario.points.Length;
			}
			Assert.That(total, Is.EqualTo(69030));
			TestContext.WriteLine("15 cases / 69030 GPU points; backend=" + SystemInfo.graphicsDeviceType + "; device=" + SystemInfo.graphicsDeviceName);
		}

		[Test]
		[Category("HairBoundaryGpuCorrectness")]
		public void QueriesUseActualProductionSubstepShapes()
		{
			var scenario = MakeScenario(103, Variation.Ordinary, 41881);
			var source = (Shape[])scenario.shapes.Clone();
			var previous = (Shape[])source.Clone();
			var next = (Shape[])source.Clone();
			var translation = new Vector3(12, -4, 7);
			for (int i = 0; i < scenario.capsuleCount; i++)
			{
				previous[i].pA -= translation;
				previous[i].pB -= translation;
				next[i].pA += translation;
				next[i].pB += translation;
				next[i].radius *= 1.5f;
			}

			int kernel = m_volume.FindKernel("KBoundariesSubstep");
			using (var shapesPrev = new ComputeBuffer(Capacity, 32))
			using (var shapesNext = new ComputeBuffer(Capacity, 32))
			using (var matrixNext = new ComputeBuffer(Capacity, 64))
			using (var matrixPrevA = new ComputeBuffer(Capacity, 64))
			using (var matrixPrevQ = new ComputeBuffer(Capacity, 16))
			using (var matrix = new ComputeBuffer(Capacity, 64))
			using (var matrixInvStep = new ComputeBuffer(Capacity, 64))
			{
				shapesPrev.SetData(previous);
				shapesNext.SetData(next);
				matrixNext.SetData(IdentityMatrices());
				matrixPrevA.SetData(IdentityMatrices());
				var rotations = new Vector4[Capacity];
				for (int i = 0; i < Capacity; i++) rotations[i] = new Vector4(0, 0, 0, 1);
				matrixPrevQ.SetData(rotations);
				m_environment.SetData(new[] { scenario.environment });

				float previousFraction = 0;
				foreach (float fraction in new[] { 0f, 0.5f, 1f })
				{
					for (int i = 0; i < scenario.capsuleCount; i++)
					{
						scenario.shapes[i].pA = Vector3.Lerp(previous[i].pA, next[i].pA, fraction);
						scenario.shapes[i].pB = Vector3.Lerp(previous[i].pB, next[i].pB, fraction);
						scenario.shapes[i].radius = Mathf.Lerp(previous[i].radius, next[i].radius, fraction);
					}
					scenario.name = "substep-" + fraction;
					scenario.points = MakePoints(scenario.shapes, 0, scenario.capsuleCount, new System.Random(41881), 0);
					using (var commands = new CommandBuffer())
					{
						BindEnvironment(commands, m_volume);
						commands.SetComputeFloatParam(m_volume, "_SubstepFractionLo", previousFraction);
						commands.SetComputeFloatParam(m_volume, "_SubstepFractionHi", fraction);
						commands.SetComputeBufferParam(m_volume, kernel, "_BoundaryShapePrev", shapesPrev);
						commands.SetComputeBufferParam(m_volume, kernel, "_BoundaryShapeNext", shapesNext);
						commands.SetComputeBufferParam(m_volume, kernel, "_BoundaryShape", m_shapes);
						commands.SetComputeBufferParam(m_volume, kernel, "_BoundaryMatrixNext", matrixNext);
						commands.SetComputeBufferParam(m_volume, kernel, "_BoundaryMatrixPrevA", matrixPrevA);
						commands.SetComputeBufferParam(m_volume, kernel, "_BoundaryMatrixPrevQ", matrixPrevQ);
						commands.SetComputeBufferParam(m_volume, kernel, "_BoundaryMatrix", matrix);
						commands.SetComputeBufferParam(m_volume, kernel, "_BoundaryMatrixInv", m_matricesInv);
						commands.SetComputeBufferParam(m_volume, kernel, "_BoundaryMatrixInvStep", matrixInvStep);
						commands.DispatchCompute(m_volume, kernel, (Capacity + 63) / 64, 1, 1);
						Validate(commands, scenario);
					}
					var actual = new Shape[Capacity];
					m_shapes.GetData(actual);
					for (int i = 0; i < scenario.capsuleCount; i++)
					{
						Assert.That(Vector3.Distance(actual[i].pA, scenario.shapes[i].pA), Is.LessThan(3e-6f), scenario.name);
						Assert.That(Vector3.Distance(actual[i].pB, scenario.shapes[i].pB), Is.LessThan(3e-6f), scenario.name);
						Assert.That(actual[i].radius, Is.EqualTo(scenario.shapes[i].radius).Within(1e-6f), scenario.name);
					}
					previousFraction = fraction;
				}
			}
		}

		void Upload(Scenario scenario)
		{
			m_shapes.SetData(scenario.shapes);
			m_environment.SetData(new[] { scenario.environment });
		}

		void BindEnvironment(CommandBuffer commands, ComputeShader shader)
		{
			commands.SetComputeConstantBufferParam(shader, m_environmentID, m_environment, 0, m_environmentBytes);
		}

		void BindQueries(CommandBuffer commands, int kernel, ComputeBuffer input, ComputeBuffer output, int count)
		{
			BindEnvironment(commands, m_queries);
			commands.SetComputeBufferParam(m_queries, kernel, "_BoundaryShape", m_shapes);
			commands.SetComputeBufferParam(m_queries, kernel, "_BoundaryMatrixInv", m_matricesInv);
			commands.SetComputeTextureParam(m_queries, kernel, "_BoundarySDF", m_sdf);
			commands.SetComputeBufferParam(m_queries, kernel, "_ValidationPoints", input);
			commands.SetComputeBufferParam(m_queries, kernel, "_ValidationResults", output);
			commands.SetComputeIntParam(m_queries, "_ValidationPointCount", count);
		}

		void Validate(CommandBuffer commands, Scenario scenario)
		{
			using (var input = new ComputeBuffer(scenario.points.Length, 12))
			using (var output = new ComputeBuffer(scenario.points.Length * ResultsPerPoint, 16))
			{
				input.SetData(scenario.points);
				BindQueries(commands, m_validateKernel, input, output, scenario.points.Length);
				commands.DispatchCompute(m_queries, m_validateKernel, (scenario.points.Length + 63) / 64, 1, 1);
				Graphics.ExecuteCommandBuffer(commands);
				var results = new Vector4[scenario.points.Length * ResultsPerPoint];
				output.GetData(results);
				int selectedKinds = 0;
				bool hasInterior = false;
				for (int i = 0; i < scenario.points.Length; i++)
				{
					int offset = ResultsPerPoint * i;
					var distance = results[offset];
					var offsets = results[offset + 1];
					var referenceOffsets = results[offset + 2];
					var normal = results[offset + 3];
					var referenceNormal = results[offset + 4];
					var bounded = results[offset + 5];
					float normalError = Vector3.Distance(normal, referenceNormal);
					if (!Finite(distance.x) || distance.x != distance.y || distance.z != distance.w ||
						referenceOffsets.w != distance.y || normal.w != distance.y ||
						!EqualXYZ(offsets, referenceOffsets) || !EqualXYZ(results[offset + 6], results[offset + 7]) ||
						referenceNormal.w != bounded.x || bounded.w != bounded.x || bounded.y != bounded.z ||
						!Finite(normalError) || normalError > 2e-6f)
					{
						Assert.Fail(scenario.name + " point=" + i + " p=" + scenario.points[i].ToString("R") +
							" distance/reference/selected/reference=" + distance.ToString("R") + " offsets=" + offsets.ToString("R") +
							" reference offsets=" + referenceOffsets.ToString("R") + " bounded=" + bounded.ToString("R") +
							" normal error=" + normalError);
					}
					if (scenario.variation == Variation.Ties && distance.x < 1e7f && distance.z != scenario.capsuleCount - 1)
						Assert.Fail("Equal capsule distances must select the highest original boundary index.");
					hasInterior |= distance.x < 0;
					selectedKinds |= 1 << SelectedKind((uint)distance.z, scenario.environment);
				}
				if (scenario.capsuleCount > 0) Assert.That(hasInterior, Is.True, scenario.name + " must exercise negative distances.");
				if (scenario.variation == Variation.Mixed)
					Assert.That(selectedKinds, Is.EqualTo(31), "All five primitive kinds must actually win mixed-set queries.");
			}
		}

		static int SelectedKind(uint selected, HairSim.VolumeCBufferEnvironment environment)
		{
			if (selected < environment._BoundaryDelimDiscrete) return 0;
			if (selected < environment._BoundaryDelimCapsule) return 1;
			if (selected < environment._BoundaryDelimSphere) return 2;
			if (selected < environment._BoundaryDelimTorus) return 3;
			return 4;
		}

		static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
		static bool EqualXYZ(Vector4 a, Vector4 b) => a.x == b.x && a.y == b.y && a.z == b.z;
		static float Next(System.Random random, float lo, float hi) => (float)(lo + (hi - lo) * random.NextDouble());
		static Vector3 Point(System.Random random, float scale) => new Vector3(Next(random, -scale, scale), Next(random, -scale, scale), Next(random, -scale, scale));

		static Matrix4x4[] IdentityMatrices()
		{
			var matrices = new Matrix4x4[Capacity];
			for (int i = 0; i < matrices.Length; i++) matrices[i] = Matrix4x4.identity;
			return matrices;
		}

		static Scenario MakeScenario(int count, Variation variation, int seed, int extraPoints = 0)
		{
			var random = new System.Random(seed);
			var shapes = new Shape[Capacity];
			bool mixed = variation == Variation.Mixed;
			int first = mixed ? 1 : 0;
			if (mixed) shapes[0].radius = 1;
			for (int i = 0; i < count; i++)
			{
				Vector3 a = mixed ? new Vector3(0, 0, -0.8f) + Point(random, 0.08f) : Point(random, 1);
				// Positive y length prevents accidental zero segments. Keep the first
				// capsule solid even for count 1; later entries still cover zero radii.
				shapes[first + i] = new Shape
				{
					pA = a,
					pB = a + new Vector3(Next(random, -0.2f, 0.2f), Next(random, 0.1f, 0.4f), Next(random, -0.2f, 0.2f)),
					radius = i % 13 == 12 ? 0 : Next(random, 0.001f, mixed ? 0.03f : 0.35f),
				};
				if (variation == Variation.Ties)
					shapes[first + i] = new Shape { pA = new Vector3(0, -0.2f, 0), pB = new Vector3(0, 0.2f, 0), radius = 0.3f };
			}
			if (variation == Variation.NegativeRadius) shapes[first].radius = -0.1f;
			if (variation == Variation.Extreme) shapes[first].pA = new Vector3(1.1e18f, 0, 0);
			if (variation == Variation.Transformed)
			{
				var rotation = Quaternion.Euler(37, 91, -24);
				for (int i = first; i < first + count; i++)
				{
					shapes[i].pA = rotation * (shapes[i].pA * 1.2f) + new Vector3(0.3f, -0.2f, 0.1f);
					shapes[i].pB = rotation * (shapes[i].pB * 1.2f) + new Vector3(0.3f, -0.2f, 0.1f);
					shapes[i].radius *= 0.8f;
				}
			}
			int capsuleEnd = first + count;
			int sphereEnd = capsuleEnd, torusEnd = capsuleEnd, end = capsuleEnd;
			if (mixed)
			{
				shapes[sphereEnd++] = new Shape { pA = new Vector3(0.7f, 0, 0), radius = 0.4f };
				torusEnd = sphereEnd;
				shapes[torusEnd++] = new Shape { pA = new Vector3(-0.7f, 0, 0), pB = Vector3.up, radius = 0.3f, extra = 0.05f };
				end = torusEnd;
				shapes[end++] = new Shape { pB = new Vector3(0.2f, 0.3f, 0.4f) };
			}
			var points = MakePoints(shapes, first, count, random, extraPoints);
			if (mixed)
			{
				points[0] = new Vector3(1.5f, 1.5f, 1.5f); // discrete wins
				points[1] = new Vector3(0.7f, 0, 0); // sphere wins
				points[2] = new Vector3(-0.4f, 0, 0); // torus wins
				points[3] = Vector3.zero; // cube wins with identity inverse transform
				points[4] = (shapes[2].pA + shapes[2].pB) * 0.5f; // capsule wins
			}
			return new Scenario
			{
				name = variation + "-" + count + "-" + seed,
				shapes = shapes,
				points = points,
				capsuleCount = count,
				variation = variation,
				environment = new HairSim.VolumeCBufferEnvironment
				{
					_BoundaryDelimDiscrete = (uint)first,
					_BoundaryDelimCapsule = (uint)capsuleEnd,
					_BoundaryDelimSphere = (uint)sphereEnd,
					_BoundaryDelimTorus = (uint)torusEnd,
					_BoundaryDelimCube = (uint)end,
					_BoundaryWorldEpsilon = Epsilon,
				},
			};
		}

		static Vector3[] MakePoints(Shape[] shapes, int first, int count, System.Random random, int extraPoints)
		{
			var points = new List<Vector3>(4098 + extraPoints + 6 * count);
			for (int i = 0; i < 4096 + extraPoints; i++) points.Add(Point(random, 1.7f));
			points.Add(Vector3.zero);
			points.Add(new Vector3(1.1e18f, 0, 0));
			for (int i = first; i < first + count; i++)
			{
				var shape = shapes[i];
				points.Add(shape.pA);
				points.Add(shape.pB);
				points.Add((shape.pA + shape.pB) * 0.5f);
				points.Add(shape.pA + Vector3.right * shape.radius);
				points.Add(shape.pA + Vector3.right * (shape.radius + 1e-5f));
				points.Add(shape.pA + Vector3.right * (shape.radius - 1e-5f));
			}
			return points.ToArray();
		}
	}
}
