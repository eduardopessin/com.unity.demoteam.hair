using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace Unity.DemoTeam.Hair.Tests
{
	public class HairCapsuleDistanceTests
	{
		const string ShaderGuid = "f9278f15d62f48a2ac3705a960d59eb4";

		static IEnumerable<TestCaseData> Cases()
		{
			var center = new Vector3(1.0f, -2.0f, 3.0f);
			yield return Case("SphereCenter", center, center, center, 2.0f);
			yield return Case("SphereInside", center, center, center + new Vector3(0.5f, 0.0f, 0.0f), 2.0f);
			yield return Case("SphereSurface", center, center, center + new Vector3(0.0f, 2.0f, 0.0f), 2.0f);
			yield return Case("SphereOutside", center, center, center + new Vector3(3.0f, 4.0f, 0.0f), 2.0f);
			yield return Case("ZeroRadiusCenter", center, center, center, 0.0f);
			yield return Case("ZeroRadiusOutside", center, center, center + new Vector3(0.0f, 0.0f, -0.5f), 0.0f);

			var a = Vector3.zero;
			var b = new Vector3(0.0f, 2.0f, 0.0f);
			yield return Case("SegmentEndpointA", a, b, a, 0.5f);
			yield return Case("SegmentEndpointB", a, b, b, 0.5f);
			yield return Case("SegmentBeforeA", a, b, new Vector3(0.0f, -1.0f, 0.0f), 0.5f);
			yield return Case("SegmentAfterB", a, b, new Vector3(0.0f, 3.0f, 0.0f), 0.5f);
			yield return Case("SegmentInside", a, b, new Vector3(0.25f, 1.0f, 0.0f), 0.5f);
			yield return Case("SegmentSurface", a, b, new Vector3(0.5f, 1.0f, 0.0f), 0.5f);
			yield return Case("SegmentOutside", a, b, new Vector3(3.0f, 1.0f, 4.0f), 0.5f);
			yield return Case("SegmentZeroRadius", a, b, new Vector3(0.0f, 1.0f, -0.25f), 0.0f);
			yield return Case("ObliqueSegment", new Vector3(-1.0f, 2.0f, 0.0f), new Vector3(2.0f, -1.0f, 3.0f), new Vector3(2.0f, 1.0f, -1.0f), 0.25f);

			// Squared length is 1e-10: an epsilon denominator would change this capsule.
			var shortB = new Vector3(0.0f, 1e-5f, 0.0f);
			yield return Case("ShortSegmentInterior", a, shortB, new Vector3(2e-6f, 5e-6f, 0.0f), 1e-6f);
			yield return Case("ShortSegmentAfterB", a, shortB, new Vector3(0.0f, 12e-6f, 0.0f), 1e-6f);
		}

		static TestCaseData Case(string name, Vector3 a, Vector3 b, Vector3 p, float radius)
		{
			return new TestCaseData(a, b, p, radius).SetName("CapsuleDistance_" + name);
		}

		[TestCaseSource(nameof(Cases))]
		public void ProductionCapsuleMatchesGeometry(Vector3 a, Vector3 b, Vector3 p, float radius)
		{
			if (!SystemInfo.supportsComputeShaders)
				Assert.Ignore("A graphics device with compute shader support is required.");

			var shader = AssetDatabase.LoadAssetAtPath<ComputeShader>(AssetDatabase.GUIDToAssetPath(ShaderGuid));
			Assert.That(shader, Is.Not.Null, "Capsule compute fixture must be imported.");
			var kernel = shader.FindKernel("EvaluateCapsule");
			var results = new Vector4[3];
			using (var buffer = new ComputeBuffer(results.Length, sizeof(float) * 4))
			{
				shader.SetVector("_PointRadius", new Vector4(p.x, p.y, p.z, radius));
				shader.SetVector("_CenterA", new Vector4(a.x, a.y, a.z, 0.0f));
				shader.SetVector("_CenterB", new Vector4(b.x, b.y, b.z, 0.0f));
				shader.SetBuffer(kernel, "_Results", buffer);
				shader.Dispatch(kernel, 1, 1, 1);
				buffer.GetData(results);
			}

			// Independent double-precision closest-point geometry, not a shader echo.
			double ax = (double)p.x - a.x, ay = (double)p.y - a.y, az = (double)p.z - a.z;
			double bx = (double)b.x - a.x, by = (double)b.y - a.y, bz = (double)b.z - a.z;
			double squaredLength = bx * bx + by * by + bz * bz;
			double t = squaredLength == 0.0 ? 0.0 : Math.Max(0.0, Math.Min(1.0, (ax * bx + ay * by + az * bz) / squaredLength));
			double x = ax - t * bx, y = ay - t * by, z = az - t * bz;
			double length = Math.Sqrt(x * x + y * y + z * z);
			double expected = length - radius;
			double tolerance = Math.Max(1e-10, Math.Max(length, radius) * 2e-5);
			AssertFiniteNear(results[0].x, expected, tolerance, "SdCapsule");
			AssertFiniteNear(results[1].x, expected, tolerance, "SdgCapsule distance");

			// The derivative is undefined on the segment (including the sphere center).
			// Only the distances above are asserted there; no fallback normal is prescribed.
			if (length > 0.0)
			{
				AssertFiniteNear(results[1].y, x / length, 2e-5, "Capsule gradient x");
				AssertFiniteNear(results[1].z, y / length, 2e-5, "Capsule gradient y");
				AssertFiniteNear(results[1].w, z / length, 2e-5, "Capsule gradient z");
			}

			if (squaredLength == 0.0)
			{
				// Sphere expectations come from CPU geometry, with production sphere parity
				// checked as well so both boundary representations remain consistent.
				AssertFiniteNear(results[0].y, expected, tolerance, "SdSphere");
				AssertFiniteNear(results[2].x, expected, tolerance, "SdgSphere distance");
				if (length > 0.0)
				{
					AssertFiniteNear(results[2].y, x / length, 2e-5, "Sphere gradient x");
					AssertFiniteNear(results[2].z, y / length, 2e-5, "Sphere gradient y");
					AssertFiniteNear(results[2].w, z / length, 2e-5, "Sphere gradient z");
					for (int component = 1; component < 4; component++)
						AssertFiniteNear(results[1][component], results[2][component], 2e-5, "Capsule/sphere gradient parity");
				}
			}
		}

		static void AssertFiniteNear(float actual, double expected, double tolerance, string label)
		{
			Assert.That(float.IsNaN(actual) || float.IsInfinity(actual), Is.False, label + " must be finite");
			Assert.That((double)actual, Is.EqualTo(expected).Within(tolerance), label);
		}
	}
}
