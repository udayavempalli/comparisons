#!/usr/bin/env python3
"""
Standalone Image Comparison Script with Automatic Anchoring
Usage: python standalone_image_compare.py <base_image> <actual_image> [--no-visual]

Dependencies: pip install opencv-python scikit-image numpy
"""

import argparse
import sys
import os
import time
import json
import cv2
import numpy as np
from skimage.metrics import structural_similarity

class StandaloneImageComparator:
    def __init__(self):
        pass
    
    def compare_images(self, base_image_path, actual_image_path, save_visual_output=True):
        """
        Compare two images using automatic anchor detection and alignment.
        Returns similarity percentage and detailed analysis.
        """
        start_time = time.time()
        
        try:
            # Load images
            img_base = cv2.imread(base_image_path)
            img_actual = cv2.imread(actual_image_path)
            
            if img_base is None or img_actual is None:
                print(f"ERROR: Could not load images")
                print(f"Base: {base_image_path} ({'Found' if img_base is not None else 'Not found'})")
                print(f"Actual: {actual_image_path} ({'Found' if img_actual is not None else 'Not found'})")
                return {"error": "Could not load images", "time_taken_ms": 0, "status": "error"}
            
            print(f"✓ Loaded images successfully")
            print(f"  Base: {img_base.shape} - {os.path.basename(base_image_path)}")
            print(f"  Actual: {img_actual.shape} - {os.path.basename(actual_image_path)}")
            
            # Convert to grayscale
            gray_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
            gray_actual = cv2.cvtColor(img_actual, cv2.COLOR_BGR2GRAY)
            
            # Resize to same dimensions for comparison
            target_height = min(img_base.shape[0], img_actual.shape[0])
            target_width = min(img_base.shape[1], img_actual.shape[1])
            
            img_base_resized = cv2.resize(img_base, (target_width, target_height))
            img_actual_resized = cv2.resize(img_actual, (target_width, target_height))
            gray_base_resized = cv2.resize(gray_base, (target_width, target_height))
            gray_actual_resized = cv2.resize(gray_actual, (target_width, target_height))
            
            print(f"✓ Resized to common size: {target_width}x{target_height}")
            
            # Initialize variables
            anchor_points_base = []
            anchor_points_actual = []
            alignment_method = "none"
            img_base_aligned = img_base_resized
            gray_base_aligned = gray_base_resized
            
            # Try automatic anchor detection and alignment
            try:
                print("🔍 Detecting anchor points...")
                
                # Method 1: Corner detection for structural elements
                corners_base = cv2.goodFeaturesToTrack(
                    gray_base_resized, maxCorners=200, qualityLevel=0.01,
                    minDistance=30, blockSize=15
                )
                corners_actual = cv2.goodFeaturesToTrack(
                    gray_actual_resized, maxCorners=200, qualityLevel=0.01,
                    minDistance=30, blockSize=15
                )
                
                corner_matches = []
                if corners_base is not None and corners_actual is not None:
                    corner_matches = self._match_corners_by_template(
                        gray_base_resized, gray_actual_resized, corners_base, corners_actual
                    )
                    print(f"  Corner matching: {len(corner_matches)} matches")
                
                # Method 2: ORB feature detection
                orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.3, nlevels=6, 
                                   edgeThreshold=20, patchSize=40)
                kp1_orb, des1_orb = orb.detectAndCompute(gray_base_resized, None)
                kp2_orb, des2_orb = orb.detectAndCompute(gray_actual_resized, None)
                
                orb_matches = []
                if des1_orb is not None and des2_orb is not None and len(kp1_orb) > 0 and len(kp2_orb) > 0:
                    # Filter strong keypoints
                    kp1_orb = [kp for kp in kp1_orb if kp.response > 0.001]
                    kp2_orb = [kp for kp in kp2_orb if kp.response > 0.001]
                    
                    if len(kp1_orb) > 0 and len(kp2_orb) > 0:
                        kp1_orb, des1_orb = orb.compute(gray_base_resized, kp1_orb)
                        kp2_orb, des2_orb = orb.compute(gray_actual_resized, kp2_orb)
                        
                        if des1_orb is not None and des2_orb is not None:
                            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                            orb_matches = bf.match(des1_orb, des2_orb)
                            orb_matches = sorted(orb_matches, key=lambda x: x.distance)
                            orb_matches = [m for m in orb_matches if m.distance < 50]
                            print(f"  ORB matching: {len(orb_matches)} quality matches")
                
                # Method 3: SIFT feature detection (if available)
                sift_matches = []
                try:
                    sift = cv2.SIFT_create(nfeatures=300, contrastThreshold=0.08, 
                                         edgeThreshold=15, sigma=2.0)
                    kp1_sift, des1_sift = sift.detectAndCompute(gray_base_resized, None)
                    kp2_sift, des2_sift = sift.detectAndCompute(gray_actual_resized, None)
                    
                    if des1_sift is not None and des2_sift is not None:
                        bf_sift = cv2.BFMatcher()
                        matches_sift = bf_sift.knnMatch(des1_sift, des2_sift, k=2)
                        for match_pair in matches_sift:
                            if len(match_pair) == 2:
                                m, n = match_pair
                                if m.distance < 0.6 * n.distance:
                                    sift_matches.append(m)
                        print(f"  SIFT matching: {len(sift_matches)} quality matches")
                except Exception as e:
                    print(f"  SIFT not available: {e}")
                
                # Choose best matching method
                best_matches = []
                kp1, kp2 = None, None
                
                if len(corner_matches) > 8:
                    best_matches = corner_matches
                    alignment_method = "CORNERS"
                    print("✓ Using corner-based alignment")
                elif len(sift_matches) > 10:
                    best_matches = sift_matches
                    kp1, kp2 = kp1_sift, kp2_sift
                    alignment_method = "SIFT"
                    print("✓ Using SIFT feature alignment")
                elif len(orb_matches) > 8:
                    best_matches = orb_matches
                    kp1, kp2 = kp1_orb, kp2_orb
                    alignment_method = "ORB"
                    print("✓ Using ORB feature alignment")
                
                # Apply alignment if we have good matches
                if len(best_matches) > 8:
                    print(f"🎯 Attempting alignment with {len(best_matches)} matches...")
                    
                    # Get point coordinates
                    if alignment_method == "CORNERS":
                        src_pts = np.float32([match[0] for match in best_matches[:50]]).reshape(-1, 1, 2)
                        dst_pts = np.float32([match[1] for match in best_matches[:50]]).reshape(-1, 1, 2)
                    else:
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in best_matches[:50]]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in best_matches[:50]]).reshape(-1, 1, 2)
                    
                    # Find homography
                    homography, mask = cv2.findHomography(
                        src_pts, dst_pts, cv2.RANSAC, 
                        ransacReprojThreshold=2.0, maxIters=10000, confidence=0.99
                    )
                    
                    # Validate homography quality
                    should_apply_alignment = False
                    if homography is not None and mask is not None:
                        inlier_indices = np.where(mask.ravel() == 1)[0]
                        inlier_ratio = len(inlier_indices) / len(src_pts)
                        det = np.linalg.det(homography[:2, :2])
                        
                        # Quality thresholds
                        if (len(inlier_indices) >= 8 and inlier_ratio >= 0.3 and 
                            0.3 <= abs(det) <= 3.0):
                            should_apply_alignment = True
                            
                            # Store good anchor points
                            inlier_src_pts = src_pts[inlier_indices]
                            inlier_dst_pts = dst_pts[inlier_indices]
                            num_anchors = min(15, len(inlier_indices))
                            anchor_points_base = [(int(pt[0][0]), int(pt[0][1])) for pt in inlier_src_pts[:num_anchors]]
                            anchor_points_actual = [(int(pt[0][0]), int(pt[0][1])) for pt in inlier_dst_pts[:num_anchors]]
                            
                            print(f"✓ Alignment validated: {len(inlier_indices)} inliers ({inlier_ratio:.1%})")
                        else:
                            print(f"⚠ Alignment rejected: {len(inlier_indices)} inliers ({inlier_ratio:.1%}), det={det:.3f}")
                    
                    if should_apply_alignment:
                        img_base_aligned = cv2.warpPerspective(img_base_resized, homography, (target_width, target_height))
                        gray_base_aligned = cv2.cvtColor(img_base_aligned, cv2.COLOR_BGR2GRAY)
                        alignment_method += "_aligned"
                        print("✓ Applied reliable alignment")
                    else:
                        alignment_method += "_direct"
                        print("⚠ Using direct comparison - alignment not reliable")
                else:
                    alignment_method = "insufficient_matches"
                    print("⚠ Not enough matches - using direct comparison")
                    
            except Exception as e:
                print(f"⚠ Anchor detection failed: {e}")
                alignment_method = "error"
            
            # Calculate similarity using SSIM
            print("📊 Calculating similarity...")
            ssim_score, diff_image = structural_similarity(gray_base_aligned, gray_actual_resized, full=True)
            similarity_percentage = ssim_score * 100
            
            # Calculate timing
            end_time = time.time()
            time_taken_seconds = end_time - start_time
            time_taken_ms = time_taken_seconds * 1000
            
            # Print results
            print(f"\n{'='*50}")
            print(f"🎯 COMPARISON RESULTS")
            print(f"{'='*50}")
            print(f"Similarity: {similarity_percentage:.2f}%")
            print(f"Method: {alignment_method}")
            print(f"Anchors: {len(anchor_points_base)} points")
            print(f"Time: {time_taken_ms:.1f}ms")
            print(f"Size: {target_width}x{target_height}")
            
            # Generate visual output if requested
            output_files = {}
            if save_visual_output:
                output_files = self._save_visual_output(
                    img_base, img_actual, img_base_aligned, img_actual_resized,
                    anchor_points_base, anchor_points_actual, diff_image,
                    similarity_percentage, alignment_method, target_width, target_height,
                    base_image_path, actual_image_path
                )
            
            return {
                "similarity_percentage": round(similarity_percentage, 2),
                "time_taken_seconds": round(time_taken_seconds, 4),
                "time_taken_ms": round(time_taken_ms, 2),
                "alignment_method": alignment_method,
                "anchor_points_detected": len(anchor_points_base),
                "comparison_size": f"{target_width}x{target_height}",
                "output_files": output_files,
                "status": "success"
            }
            
        except Exception as e:
            end_time = time.time()
            time_taken_ms = (end_time - start_time) * 1000
            print(f"❌ Error: {str(e)}")
            return {
                "error": str(e),
                "time_taken_ms": round(time_taken_ms, 2),
                "status": "error"
            }
    
    def _match_corners_by_template(self, img1, img2, corners1, corners2):
        """Match corners using template matching"""
        matches = []
        template_size = 40
        
        if corners1 is None or corners2 is None:
            return matches
            
        corners1 = corners1.reshape(-1, 2)
        corners2 = corners2.reshape(-1, 2)
        
        for corner1 in corners1:
            x1, y1 = int(corner1[0]), int(corner1[1])
            
            if (x1 - template_size//2 >= 0 and y1 - template_size//2 >= 0 and 
                x1 + template_size//2 < img1.shape[1] and y1 + template_size//2 < img1.shape[0]):
                
                template = img1[y1-template_size//2:y1+template_size//2, 
                              x1-template_size//2:x1+template_size//2]
                
                best_match_val = -1
                best_match_corner = None
                
                for corner2 in corners2:
                    x2, y2 = int(corner2[0]), int(corner2[1])
                    
                    if (x2 - template_size//2 >= 0 and y2 - template_size//2 >= 0 and 
                        x2 + template_size//2 < img2.shape[1] and y2 + template_size//2 < img2.shape[0]):
                        
                        search_region = img2[y2-template_size//2:y2+template_size//2, 
                                           x2-template_size//2:x2+template_size//2]
                        
                        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(result)
                        
                        if max_val > 0.7 and max_val > best_match_val:
                            best_match_val = max_val
                            best_match_corner = corner2
                
                if best_match_corner is not None and best_match_val > 0.75:
                    matches.append((corner1, best_match_corner))
        
        return matches
    
    def _save_visual_output(self, img_base, img_actual, img_base_aligned, img_actual_resized,
                           anchor_points_base, anchor_points_actual, diff_image,
                           similarity_percentage, alignment_method, target_width, target_height,
                           base_image_path, actual_image_path):
        """Save visual comparison outputs"""
        output_dir = "comparison_output"
        os.makedirs(output_dir, exist_ok=True)
        output_files = {}
        
        try:
            # 1. Original images side by side
            original_comparison = np.hstack([
                cv2.resize(img_base, (400, 300)),
                cv2.resize(img_actual, (400, 300))
            ])
            cv2.putText(original_comparison, "Base Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(original_comparison, "Actual Image", (410, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            original_path = os.path.join(output_dir, "01_original_images.jpg")
            cv2.imwrite(original_path, original_comparison)
            output_files["original_images"] = original_path
            
            # 2. Anchor points visualization
            if anchor_points_base and anchor_points_actual:
                anchor_base_vis = img_base_aligned.copy()
                anchor_actual_vis = img_actual_resized.copy()
                
                for i, (x, y) in enumerate(anchor_points_base):
                    cv2.circle(anchor_base_vis, (x, y), 10, (0, 255, 0), 3)
                    cv2.putText(anchor_base_vis, str(i+1), (x+12, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                for i, (x, y) in enumerate(anchor_points_actual):
                    cv2.circle(anchor_actual_vis, (x, y), 10, (0, 255, 0), 3)
                    cv2.putText(anchor_actual_vis, str(i+1), (x+12, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                anchor_comparison = np.hstack([anchor_base_vis, anchor_actual_vis])
                cv2.putText(anchor_comparison, f"Anchor Points ({alignment_method})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(anchor_comparison, f"{len(anchor_points_base)} reliable anchors", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                anchor_path = os.path.join(output_dir, "02_anchor_points.jpg")
                cv2.imwrite(anchor_path, anchor_comparison)
                output_files["anchor_points"] = anchor_path
            
            # 3. Aligned comparison
            aligned_comparison = np.hstack([img_base_aligned, img_actual_resized])
            cv2.putText(aligned_comparison, f"Aligned ({alignment_method})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(aligned_comparison, "Actual", (target_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            aligned_path = os.path.join(output_dir, "03_aligned_comparison.jpg")
            cv2.imwrite(aligned_path, aligned_comparison)
            output_files["aligned_comparison"] = aligned_path
            
            # 4. Difference heatmap
            diff_normalized = (diff_image * 255).astype(np.uint8)
            diff_colored = cv2.applyColorMap(255 - diff_normalized, cv2.COLORMAP_JET)
            
            diff_path = os.path.join(output_dir, "04_difference_heatmap.jpg")
            cv2.imwrite(diff_path, diff_colored)
            output_files["difference_heatmap"] = diff_path
            
            # 5. Overlay
            overlay = cv2.addWeighted(img_base_aligned, 0.5, img_actual_resized, 0.5, 0)
            cv2.putText(overlay, f"Similarity: {similarity_percentage:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            overlay_path = os.path.join(output_dir, "05_overlay.jpg")
            cv2.imwrite(overlay_path, overlay)
            output_files["overlay"] = overlay_path
            
            print(f"\n📁 Visual outputs saved to: {os.path.abspath(output_dir)}")
            for name, path in output_files.items():
                print(f"   {name}: {os.path.basename(path)}")
                
        except Exception as e:
            print(f"⚠ Warning: Could not save visual outputs: {e}")
        
        return output_files


def main():
    parser = argparse.ArgumentParser(
        description='Compare two images using automatic anchor detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python standalone_image_compare.py image1.jpg image2.jpg
  python standalone_image_compare.py base.png actual.png --no-visual
  python standalone_image_compare.py img1.jpg img2.jpg --json
        """
    )
    
    parser.add_argument('base_image', help='Path to the base/reference image')
    parser.add_argument('actual_image', help='Path to the actual/comparison image')
    parser.add_argument('--no-visual', action='store_true', 
                       help='Skip generating visual output files')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.base_image):
        print(f"❌ Base image not found: {args.base_image}")
        sys.exit(1)
    
    if not os.path.exists(args.actual_image):
        print(f"❌ Actual image not found: {args.actual_image}")
        sys.exit(1)
    
    # Run comparison
    print(f"🚀 Starting image comparison...")
    print(f"   Base: {args.base_image}")
    print(f"   Actual: {args.actual_image}")
    
    comparator = StandaloneImageComparator()
    result = comparator.compare_images(
        args.base_image,
        args.actual_image,
        save_visual_output=not args.no_visual
    )
    
    if args.json:
        print(f"\n📋 JSON Result:")
        print(json.dumps(result, indent=2))
    
    if result.get("status") != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()