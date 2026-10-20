using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

namespace Unity.DemoTeam.Hair.Tests
{
	public class HairBoundaryCaptureTests
	{
		struct CaptureData
		{
			public int handle;
			public Vector3 position;
		}

		// Exercise the public generic consumer, not its private buffer fields. An
		// irrelevant collider must still consume overlap capacity before filtering.
		struct CaptureProxy : ISpatialComponentProxy<HairBoundary, CaptureData>
		{
			public bool TryGetData(HairBoundary component, ref CaptureData data)
			{
				data.handle = component.GetInstanceID();
				data.position = component.transform.position;
				return true;
			}

			public bool TryGetComponentData(Component component, ref CaptureData data) => false;
			public int ResolveDataHandle(in CaptureData data) => data.handle;
			public float ResolveDataDistance(in CaptureData data, in Vector3 p) => Vector3.Distance(data.position, p);
		}

		[Test]
		[Category("HairBoundaryGpuCorrectness")]
		public void VolumeCaptureRetains256BoundariesAmongSaturatingIrrelevantColliders()
		{
			// 256 relevant + 513 irrelevant colliders force both the initial 256-slot
			// and next 512-slot overlap buffers to saturate on a fresh generic filter.
			// No dependence on Physics overlap ordering or on a project scene/asset.
			var root = new GameObject("Boundary capture test") { hideFlags = HideFlags.HideAndDontSave };
			root.transform.position = new Vector3(2731, -1452, 3294);
			var boundaries = new List<HairBoundary>();
			var expected = new HashSet<int>();
			var bounds = new Bounds(root.transform.position, Vector3.one * 10);
			LayerMask layer = 1 << 31;
			try
			{
				for (int i = 0; i < 256 + 513; i++)
				{
					var child = new GameObject("Capture collider") { layer = 31 };
					child.transform.SetParent(root.transform, false);
					child.transform.localPosition = new Vector3(i % 10, (i / 10) % 10, i / 100) * 0.1f - Vector3.one * 0.5f;
					var collider = child.AddComponent<CapsuleCollider>();
					collider.radius = 0.025f;
					collider.height = 0.1f;
					if (i < 256)
					{
						var boundary = child.AddComponent<HairBoundary>();
						boundaries.Add(boundary);
						expected.Add(boundary.GetInstanceID());
					}
				}
				Physics.SyncTransforms();
				AssertCapture(boundaries, expected, bounds, layer);
				// Reuse the retained overlap buffer; resident and volume captures must
				// deduplicate and must not leak entries from the previous gather.
				for (int i = 0; i < 103; i++)
				{
					expected.Remove(boundaries[i].GetInstanceID());
					boundaries[i].gameObject.SetActive(false);
				}
				Physics.SyncTransforms();
				AssertCapture(boundaries, expected, bounds, layer);
			}
			finally
			{
				Object.DestroyImmediate(root);
				Physics.SyncTransforms();
			}
		}

		static void AssertCapture(List<HairBoundary> boundaries, HashSet<int> expected, Bounds bounds, LayerMask layer)
		{
			// One resident overlaps the volume; all other entries come from physics.
			var residents = new[] { boundaries[boundaries.Count - 1] };
			var gathered = SpatialComponentFilter<HairBoundary, CaptureData, CaptureProxy>.Gather(residents, true, bounds, layer, true, false);
			var actual = new List<int>(gathered.Count);
			foreach (var data in gathered) actual.Add(data.handle);
			Assert.That(actual.Count, Is.EqualTo(expected.Count), "Collider saturation must not truncate filtered boundary capture.");
			CollectionAssert.AreEquivalent(expected, actual);
		}
	}
}
