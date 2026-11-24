import unittest
import numpy as np
from cropped_gait import band_edge, stable_top_band, cropped_track, background_mask


class CroppedTopBandTests(unittest.TestCase):
    def scene(self, seed, edge=54):
        rng=np.random.default_rng(seed)
        image=rng.integers(78,240,(598,960,3),dtype=np.uint8)
        if edge:
            image[:edge]=[42,27,14]
            image[9:38,14+3*seed:170+3*seed]=[205,188,162]
        return image

    def test_top_band_with_changing_text(self):
        images=[self.scene(i,54+i%2) for i in range(5)]
        self.assertIn(stable_top_band(images,598),(54,55))
        self.assertEqual(stable_top_band(images[:2],598),0)

    def test_unstable_or_unbounded_band_is_rejected(self):
        self.assertEqual(stable_top_band([self.scene(i,40+8*i) for i in range(5)],598),0)
        self.assertEqual(band_edge(self.scene(0,0)),0)
        image=np.full((598,960,3),29,np.uint8)
        self.assertEqual(band_edge(image),0)

    def test_cropped_geometry_is_relative_to_confirmed_content_edge(self):
        k=np.zeros((50,17,3)); k[:,13:17,2]=.8
        data=dict(height=598,boxes=np.tile([100,62,650,547],(50,1)),keypoints=k)
        rows=np.arange(50)
        self.assertFalse(cropped_track(data,rows,.5))
        self.assertTrue(cropped_track(data,rows,.5,54))
        k[:8,5,2]=.9
        self.assertFalse(cropped_track(data,rows,.5,54))

    def test_motion_registration_excludes_bar_only_when_necessary(self):
        mask=background_mask((300,480),[],53)
        self.assertFalse(mask[52].any())
        self.assertTrue(mask[53].all())


if __name__=='__main__': unittest.main()
