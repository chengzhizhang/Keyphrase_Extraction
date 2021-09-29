/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    DiscretizeFilter.java
 *    Copyright (C) 1999 Eibe Frank,Len Trigg
 *
 */


package weka.filters;

import java.io.*;
import java.util.*;
import weka.core.*;

/** 
 * An instance filter that discretizes a range of numeric attributes in 
 * the dataset into nominal attributes. Discretization can be either by 
 * simple binning, or by Fayyad & Irani's MDL method (the default).<p>
 *
 * Valid filter-specific options are: <p>
 *
 * -B num <br>
 * Specifies the (maximum) number of bins to divide numeric attributes into.
 * (default: class-based discretisation).<p>
 *
 * -F <br>
 * Use equal-frequency instead of equal-width discretization if 
 * class-based discretisation is turned off.<p>
 *
 * -O <br>
 * Optimize the number of bins using a leave-one-out estimate of the 
 * entropy (for equal-width binning).<p>
 *
 * -R col1,col2-col4,... <br>
 * Specifies list of columns to Discretize. First
 * and last are valid indexes. (default: none) <p>
 *
 * -V <br>
 * Invert matching sense.<p>
 *
 * -D <br>
 * Make binary nominal attributes. <p>
 *
 * -E <br>
 * Use better encoding of split point for MDL. <p>
 *   
 * -K <br>
 * Use Kononeko's MDL criterion. <p>
 * 
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 1.16 $
 */
public class DiscretizeFilter extends Filter 
  implements OptionHandler, WeightedInstancesHandler {

  /** Stores which columns to Discretize */
  protected Range m_DiscretizeCols = new Range();

  /** The number of bins to divide the attribute into */
  protected int m_NumBins = 10;

  /** Store the current cutpoints */
  protected double [][] m_CutPoints = null;

  /** True if discretisation will be done by MDL rather than binning */
  protected boolean m_UseMDL = true;

  /** Output binary attributes for discretized attributes. */
  protected boolean m_MakeBinary = false;

  /** Use better encoding of split point for MDL. */
  protected boolean m_UseBetterEncoding = false;

  /** Use Kononenko's MDL criterion instead of Fayyad et al.'s */
  protected boolean m_UseKononenko = false;

  /** Find the number of bins using cross-validated entropy. */
  protected boolean m_FindNumBins = false;

  /** Use equal-frequency binning if unsupervised discretization turned on */
  protected boolean m_UseEqualFrequency = false;

  /** Constructor - initialises the filter */
  public DiscretizeFilter() {

    setAttributeIndices("first-last");
  }


  /**
   * Gets an enumeration describing the available options
   *
   * @return an enumeration of all the available options
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(7);

    newVector.addElement(new Option(
              "\tSpecifies the (maximum) number of bins to divide numeric"
	      + " attributes into.\n"
	      + "\t(default class-based discretization)",
              "B", 1, "-B <num>"));

    newVector.addElement(new Option(
              "\tUse equal-frequency instead of equal-width with\n"+
	      "\tunsupervised discretization.",
              "F", 0, "-F"));

    newVector.addElement(new Option(
              "\tOptimize number of bins using leave-one-out estimate\n"+
	      "\tof estimated entropy (for equal-width discretization).",
              "O", 0, "-O"));

    /* If we decide to implement loading and saving cutfiles like 
     * the C Discretizer (which is probably not necessary)
    newVector.addElement(new Option(
              "\tSpecify that the cutpoints should be loaded from a file.",
              "L", 1, "-L <file>"));
    newVector.addElement(new Option(
              "\tSpecify that the chosen cutpoints should be saved to a file.",
              "S", 1, "-S <file>"));
    */

    newVector.addElement(new Option(
              "\tSpecifies list of columns to Discretize. First"
	      + " and last are valid indexes.\n"
	      + "\t(default none)",
              "R", 1, "-R <col1,col2-col4,...>"));

    newVector.addElement(new Option(
              "\tInvert matching sense of column indexes.",
              "V", 0, "-V"));

    newVector.addElement(new Option(
              "\tOutput binary attributes for discretized attributes.",
              "D", 0, "-D"));

    newVector.addElement(new Option(
              "\tUse better encoding of split point for MDL.",
              "E", 0, "-E"));

    newVector.addElement(new Option(
              "\tUse Kononenko's MDL criterion.",
              "K", 0, "-K"));

    return newVector.elements();
  }


  /**
   * Parses the options for this object. Valid options are: <p>
   *
   * -B num <br>
   * Specifies the (maximum) number of bins to divide numeric attributes into.
   * (default class-based discretisation).<p>
   *
   * -F <br>
   * Use equal-frequency instead of equal-width discretization if 
   * class-based discretisation is turned off.<p>
   *
   * -O <br>
   * Optimize the number of bins using a leave-one-out estimate of the 
   * entropy (for equal-width binning).<p>
   *
   * -R col1,col2-col4,... <br>
   * Specifies list of columns to Discretize. First
   * and last are valid indexes. (default none) <p>
   *
   * -V <br>
   * Invert matching sense.<p>
   *
   * -D <br>
   * Make binary nominal attributes. <p>
   *
   * -E <br>
   * Use better encoding of split point for MDL. <p>
   *   
   * -K <br>
   * Use Kononeko's MDL criterion. <p>
   * 
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    setMakeBinary(Utils.getFlag('D', options));
    setUseEqualFrequency(Utils.getFlag('F', options));
    setUseBetterEncoding(Utils.getFlag('E', options));
    setUseKononenko(Utils.getFlag('K', options));
    setFindNumBins(Utils.getFlag('O', options));
    setInvertSelection(Utils.getFlag('V', options));
    setUseMDL(true);

    String numBins = Utils.getOption('B', options);
    if (numBins.length() != 0) {
      setBins(Integer.parseInt(numBins));
      setUseMDL(false);
    } else {
      setBins(10);
    }
    
    String convertList = Utils.getOption('R', options);
    if (convertList.length() != 0) {
      setAttributeIndices(convertList);
    } else {
      setAttributeIndices("first-last");
    }

    if (getInputFormat() != null) {
      setInputFormat(getInputFormat());
    }
  }
  /**
   * Gets the current settings of the filter.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {

    String [] options = new String [12];
    int current = 0;

    if (getMakeBinary()) {
      options[current++] = "-D";
    }
    if (getUseEqualFrequency()) {
      options[current++] = "-F";
    }
    if (getUseBetterEncoding()) {
      options[current++] = "-E";
    }
    if (getUseKononenko()) {
      options[current++] = "-K";
    }
    if (getFindNumBins()) {
      options[current++] = "-O";
    }
    if (getInvertSelection()) {
      options[current++] = "-V";
    }
    if (!getUseMDL()) {
      options[current++] = "-B"; options[current++] = "" + getBins();
    }
    if (!getAttributeIndices().equals("")) {
      options[current++] = "-R"; options[current++] = getAttributeIndices();
    }
    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }


  /**
   * Sets the format of the input instances.
   *
   * @param instanceInfo an Instances object containing the input instance
   * structure (any instances contained in the object are ignored - only the
   * structure is required).
   * @return true if the outputFormat may be collected immediately
   * @exception Exception if the input format can't be set successfully
   */
  public boolean setInputFormat(Instances instanceInfo) throws Exception {

    super.setInputFormat(instanceInfo);

    m_DiscretizeCols.setUpper(instanceInfo.numAttributes() - 1);
    m_CutPoints = null;
    if (m_UseMDL) {
      if (instanceInfo.classIndex() < 0) {
	throw new UnassignedClassException("Cannot use class-based discretization: "
                                           + "no class assigned to the dataset");
      }
      if (!instanceInfo.classAttribute().isNominal()) {
	throw new UnsupportedClassTypeException("Supervised discretization not possible:"
                                                + " class is not nominal!");
      }
    } else {
      if (getFindNumBins() && getUseEqualFrequency()) {
	throw new IllegalArgumentException("Bin number optimization in conjunction "+
					   "with equal-frequency binning not implemented.");
      }
    }

    // If we implement loading cutfiles, then load 
    //them here and set the output format
    return false;
  }

  

  /**
   * Input an instance for filtering. Ordinarily the instance is processed
   * and made available for output immediately. Some filters require all
   * instances be read before producing output.
   *
   * @param instance the input instance
   * @return true if the filtered instance may now be
   * collected with output().
   * @exception IllegalStateException if no input format has been defined.
   */
  public boolean input(Instance instance) {

    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }
    if (m_NewBatch) {
      resetQueue();
      m_NewBatch = false;
    }
    
    if (m_CutPoints != null) {
      convertInstance(instance);
      return true;
    }

    bufferInput(instance);
    return false;
  }


  /**
   * Signifies that this batch of input to the filter is finished. If the 
   * filter requires all instances prior to filtering, output() may now 
   * be called to retrieve the filtered instances.
   *
   * @return true if there are instances pending output
   * @exception IllegalStateException if no input structure has been defined
   */
  public boolean batchFinished() {

    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }
    if (m_CutPoints == null) {
      calculateCutPoints();

      setOutputFormat();

      // If we implement saving cutfiles, save the cuts here

      // Convert pending input instances
      for(int i = 0; i < getInputFormat().numInstances(); i++) {
	convertInstance(getInputFormat().instance(i));
      }
    } 
    flushInput();

    m_NewBatch = true;
    return (numPendingOutput() != 0);
  }

  /**
   * Returns a string describing this filter
   *
   * @return a description of the filter suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {

    return "An instance filter that discretizes a range of numeric"
      + " attributes in the dataset into nominal attributes."
      + " Discretization can be either by simple binning, or by"
      + " Fayyad & Irani's MDL method (the default).";
  }
  
  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String findNumBinsTipText() {

    return "Optimize number of equal-width bins using leave-one-out.";
  }

  /**
   * Get the value of FindNumBins.
   *
   * @return Value of FindNumBins.
   */
  public boolean getFindNumBins() {
    
    return m_FindNumBins;
  }
  
  /**
   * Set the value of FindNumBins.
   *
   * @param newFindNumBins Value to assign to FindNumBins.
   */
  public void setFindNumBins(boolean newFindNumBins) {
    
    m_FindNumBins = newFindNumBins;
  }
  
  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String makeBinaryTipText() {

    return "Make resulting attributes binary.";
  }

  /**
   * Gets whether binary attributes should be made for discretized ones.
   *
   * @return true if attributes will be binarized
   */
  public boolean getMakeBinary() {

    return m_MakeBinary;
  }

  /** 
   * Sets whether binary attributes should be made for discretized ones.
   *
   * @param makeBinary if binary attributes are to be made
   */
  public void setMakeBinary(boolean makeBinary) {

    m_MakeBinary = makeBinary;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String useMDLTipText() {

    return "Use class-based discretization. If set to false, does"
      + " not require a class attribute, and uses a fixed number"
      + " of bins (according to bins setting).";
  }

  /**
   * Gets whether MDL will be used as the discretisation method.
   *
   * @return true if so, false if fixed bins should be used.
   */
  public boolean getUseMDL() {

    return m_UseMDL;
  }

  /** 
   * Sets whether MDL will be used as the discretisation method.
   *
   * @param useMDL true if MDL should be used, false if fixed bins should
   * be used.
   */
  public void setUseMDL(boolean useMDL) {

    m_UseMDL = useMDL;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String useKononenkoTipText() {

    return "Use Kononenko's MDL criterion. If set to false"
      + " uses the Fayyad & Irani criterion.";
  }
  
  /**
   * Get the value of UseEqualFrequency.
   *
   * @return Value of UseEqualFrequency.
   */
  public boolean getUseEqualFrequency() {
    
    return m_UseEqualFrequency;
  }
  
  /**
   * Set the value of UseEqualFrequency.
   *
   * @param newUseEqualFrequency Value to assign to UseEqualFrequency.
   */
  public void setUseEqualFrequency(boolean newUseEqualFrequency) {
    
    m_UseEqualFrequency = newUseEqualFrequency;
  }
  
  /**
   * Gets whether Kononenko's MDL criterion is to be used.
   *
   * @return true if Kononenko's criterion will be used.
   */
  public boolean getUseKononenko() {

    return m_UseKononenko;
  }

  /** 
   * Sets whether Kononenko's MDL criterion is to be used.
   *
   * @param useKon true if Kononenko's one is to be used
   */
  public void setUseKononenko(boolean useKon) {

    m_UseKononenko = useKon;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String useBetterEncodingTipText() {

    return "Uses a more efficient split point encoding.";
  }

  /**
   * Gets whether better encoding is to be used for MDL.
   *
   * @return true if the better MDL encoding will be used
   */
  public boolean getUseBetterEncoding() {

    return m_UseBetterEncoding;
  }

  /** 
   * Sets whether better encoding is to be used for MDL.
   *
   * @param useBetterEncoding true if better encoding to be used.
   */
  public void setUseBetterEncoding(boolean useBetterEncoding) {

    m_UseBetterEncoding = useBetterEncoding;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String binsTipText() {

    return "Number of bins for class-blind discretisation. This"
      + " setting is ignored if MDL-based discretisation is used.";
  }

  /**
   * Gets the number of bins numeric attributes will be divided into
   *
   * @return the number of bins.
   */
  public int getBins() {

    return m_NumBins;
  }

  /**
   * Sets the number of bins to divide each selected numeric attribute into
   *
   * @param numBins the number of bins
   */
  public void setBins(int numBins) {

    m_NumBins = numBins;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String invertSelectionTipText() {

    return "Set attribute selection mode. If false, only selected"
      + " (numeric) attributes in the range will be discretized; if"
      + " true, only non-selected attributes will be discretized.";
  }

  /**
   * Gets whether the supplied columns are to be removed or kept
   *
   * @return true if the supplied columns will be kept
   */
  public boolean getInvertSelection() {

    return m_DiscretizeCols.getInvert();
  }

  /**
   * Sets whether selected columns should be removed or kept. If true the 
   * selected columns are kept and unselected columns are deleted. If false
   * selected columns are deleted and unselected columns are kept.
   *
   * @param invert the new invert setting
   */
  public void setInvertSelection(boolean invert) {

    m_DiscretizeCols.setInvert(invert);
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String attributeIndicesTipText() {
    return "Specify range of attributes to act on."
      + " This is a comma separated list of attribute indices, with"
      + " \"first\" and \"last\" valid values. Specify an inclusive"
      + " range with \"-\". E.g: \"first-3,5,6-10,last\".";
  }

  /**
   * Gets the current range selection
   *
   * @return a string containing a comma separated list of ranges
   */
  public String getAttributeIndices() {

    return m_DiscretizeCols.getRanges();
  }

  /**
   * Sets which attributes are to be Discretized (only numeric
   * attributes among the selection will be Discretized).
   *
   * @param rangeList a string representing the list of attributes. Since
   * the string will typically come from a user, attributes are indexed from
   * 1. <br>
   * eg: first-3,5,6-last
   * @exception IllegalArgumentException if an invalid range list is supplied 
   */
  public void setAttributeIndices(String rangeList) {

    m_DiscretizeCols.setRanges(rangeList);
  }

  /**
   * Sets which attributes are to be Discretized (only numeric
   * attributes among the selection will be Discretized).
   *
   * @param attributes an array containing indexes of attributes to Discretize.
   * Since the array will typically come from a program, attributes are indexed
   * from 0.
   * @exception IllegalArgumentException if an invalid set of ranges
   * is supplied 
   */
  public void setAttributeIndicesArray(int [] attributes) {

    setAttributeIndices(Range.indicesToRangeList(attributes));
  }

  /**
   * Gets the cut points for an attribute
   *
   * @param the index (from 0) of the attribute to get the cut points of
   * @return an array containing the cutpoints (or null if the
   * attribute requested isn't being Discretized
   */
  public double [] getCutPoints(int attributeIndex) {

    if (m_CutPoints == null) {
      return null;
    }
    return m_CutPoints[attributeIndex];
  }

  /** Generate the cutpoints for each attribute */
  protected void calculateCutPoints() {

    Instances copy = null;

    m_CutPoints = new double [getInputFormat().numAttributes()] [];
    for(int i = getInputFormat().numAttributes() - 1; i >= 0; i--) {
      if ((m_DiscretizeCols.isInRange(i)) && 
	  (getInputFormat().attribute(i).isNumeric())) {
	if (m_UseMDL) {

	  // Use copy to preserve order
	  if (copy == null) {
	    copy = new Instances(getInputFormat());
	  }
	  calculateCutPointsByMDL(i, copy);
	} else {
	  if (m_FindNumBins) {
	    findNumBins(i);
	  } else if (!m_UseEqualFrequency) {
	    calculateCutPointsByEqualWidthBinning(i);
	  } else {
	    calculateCutPointsByEqualFrequencyBinning(i);
	  }
	}
      }
    }
  }

  /**
   * Set cutpoints for a single attribute using MDL.
   *
   * @param index the index of the attribute to set cutpoints for
   */
  protected void calculateCutPointsByMDL(int index,
					 Instances data) {

    // Sort instances
    data.sort(data.attribute(index));

    // Find first instances that's missing
    int firstMissing = data.numInstances();
    for (int i = 0; i < data.numInstances(); i++) {
      if (data.instance(i).isMissing(index)) {
        firstMissing = i;
        break;
      }
    }
    m_CutPoints[index] = cutPointsForSubset(data, index, 0, firstMissing);
  }

  /** Test using Kononenko's MDL criterion. */
  private boolean KononenkosMDL(double[] priorCounts,
				double[][] bestCounts,
				double numInstances,
				int numCutPoints) {

    double distPrior, instPrior, distAfter = 0, sum, instAfter = 0;
    double before, after;
    int numClassesTotal;

    // Number of classes occuring in the set
    numClassesTotal = 0;
    for (int i = 0; i < priorCounts.length; i++) {
      if (priorCounts[i] > 0) {
	numClassesTotal++;
      }
    }

    // Encode distribution prior to split
    distPrior = SpecialFunctions.log2Binomial(numInstances 
					      + numClassesTotal - 1,
					      numClassesTotal - 1);

    // Encode instances prior to split.
    instPrior = SpecialFunctions.log2Multinomial(numInstances,
						 priorCounts);

    before = instPrior + distPrior;

    // Encode distributions and instances after split.
    for (int i = 0; i < bestCounts.length; i++) {
      sum = Utils.sum(bestCounts[i]);
      distAfter += SpecialFunctions.log2Binomial(sum + numClassesTotal - 1,
						 numClassesTotal - 1);
      instAfter += SpecialFunctions.log2Multinomial(sum,
						    bestCounts[i]);
    }

    // Coding cost after split
    after = Utils.log2(numCutPoints) + distAfter + instAfter;

    // Check if split is to be accepted
    return (Utils.gr(before, after));
  }


  /** Test using Fayyad and Irani's MDL criterion. */
  private boolean FayyadAndIranisMDL(double[] priorCounts,
				     double[][] bestCounts,
				     double numInstances,
				     int numCutPoints) {

    double priorEntropy, entropy, gain; 
    double entropyLeft, entropyRight, delta;
    int numClassesTotal, numClassesRight, numClassesLeft;

    // Compute entropy before split.
    priorEntropy = ContingencyTables.entropy(priorCounts);

    // Compute entropy after split.
    entropy = ContingencyTables.entropyConditionedOnRows(bestCounts);

    // Compute information gain.
    gain = priorEntropy - entropy;

    // Number of classes occuring in the set
    numClassesTotal = 0;
    for (int i = 0; i < priorCounts.length; i++) {
      if (priorCounts[i] > 0) {
	numClassesTotal++;
      }
    }

    // Number of classes occuring in the left subset
    numClassesLeft = 0;
    for (int i = 0; i < bestCounts[0].length; i++) {
      if (bestCounts[0][i] > 0) {
	numClassesLeft++;
      }
    }

    // Number of classes occuring in the right subset
    numClassesRight = 0;
    for (int i = 0; i < bestCounts[1].length; i++) {
      if (bestCounts[1][i] > 0) {
	numClassesRight++;
      }
    }

    // Entropy of the left and the right subsets
    entropyLeft = ContingencyTables.entropy(bestCounts[0]);
    entropyRight = ContingencyTables.entropy(bestCounts[1]);

    // Compute terms for MDL formula
    delta = Utils.log2(Math.pow(3, numClassesTotal) - 2) - 
      (((double) numClassesTotal * priorEntropy) - 
       (numClassesRight * entropyRight) - 
       (numClassesLeft * entropyLeft));

    // Check if split is to be accepted
    return (Utils.gr(gain, (Utils.log2(numCutPoints) + delta) /
		     (double)numInstances));
  }
    

  /** Selects cutpoints for sorted subset. */
  private double[] cutPointsForSubset(Instances instances, int attIndex, 
				      int first, int lastPlusOne) { 

    double[][] counts, bestCounts;
    double[] priorCounts, left, right, cutPoints;
    double currentCutPoint = -Double.MAX_VALUE, bestCutPoint = -1, 
      currentEntropy, bestEntropy, priorEntropy, gain;
    int bestIndex = -1, numInstances = 0, numCutPoints = 0;

    // Compute number of instances in set
    if ((lastPlusOne - first) < 2) {
      return null;
    }

    // Compute class counts.
    counts = new double[2][instances.numClasses()];
    for (int i = first; i < lastPlusOne; i++) {
      numInstances += instances.instance(i).weight();
      counts[1][(int)instances.instance(i).classValue()] +=
	instances.instance(i).weight();
    }

    // Save prior counts
    priorCounts = new double[instances.numClasses()];
    System.arraycopy(counts[1], 0, priorCounts, 0, 
		     instances.numClasses());

    // Entropy of the full set
    priorEntropy = ContingencyTables.entropy(priorCounts);
    bestEntropy = priorEntropy;
    
    // Find best entropy.
    bestCounts = new double[2][instances.numClasses()];
    for (int i = first; i < (lastPlusOne - 1); i++) {
      counts[0][(int)instances.instance(i).classValue()] +=
	instances.instance(i).weight();
      counts[1][(int)instances.instance(i).classValue()] -=
	instances.instance(i).weight();
      if (Utils.sm(instances.instance(i).value(attIndex), 
		   instances.instance(i + 1).value(attIndex))) {
	currentCutPoint = instances.instance(i).value(attIndex); //+ 
	//instances.instance(i + 1).value(attIndex)) / 2.0;
	currentEntropy = ContingencyTables.entropyConditionedOnRows(counts);
	if (Utils.sm(currentEntropy, bestEntropy)) {
	  bestCutPoint = currentCutPoint;
	  bestEntropy = currentEntropy;
	  bestIndex = i;
	  System.arraycopy(counts[0], 0, 
			   bestCounts[0], 0, instances.numClasses());
	  System.arraycopy(counts[1], 0, 
			   bestCounts[1], 0, instances.numClasses()); 
	}
	numCutPoints++;
      }
    }

    // Use worse encoding?
    if (!m_UseBetterEncoding) {
      numCutPoints = (lastPlusOne - first) - 1;
    }

    // Checks if gain is zero
    gain = priorEntropy - bestEntropy;
    if (Utils.eq(gain, 0)) {
      return null;
    }

    // Check if split is to be accepted
    if ((m_UseKononenko && KononenkosMDL(priorCounts, bestCounts,
					 numInstances, numCutPoints)) ||
	(!m_UseKononenko && FayyadAndIranisMDL(priorCounts, bestCounts,
					       numInstances, numCutPoints))) {
      
      // Select split points for the left and right subsets
      left = cutPointsForSubset(instances, attIndex, first, bestIndex + 1);
      right = cutPointsForSubset(instances, attIndex, 
				 bestIndex + 1, lastPlusOne);
      
      // Merge cutpoints and return them
      if ((left == null) && (right) == null) {
	cutPoints = new double[1];
	cutPoints[0] = bestCutPoint;
      } else if (right == null) {
	cutPoints = new double[left.length + 1];
	System.arraycopy(left, 0, cutPoints, 0, left.length);
	cutPoints[left.length] = bestCutPoint;
      } else if (left == null) {
	cutPoints = new double[1 + right.length];
	cutPoints[0] = bestCutPoint;
	System.arraycopy(right, 0, cutPoints, 1, right.length);
      } else {
	cutPoints =  new double[left.length + right.length + 1];
	cutPoints = new double[left.length + right.length + 1];
	System.arraycopy(left, 0, cutPoints, 0, left.length);
	cutPoints[left.length] = bestCutPoint;
	System.arraycopy(right, 0, cutPoints, left.length + 1, right.length);
      }
      
      return cutPoints;
    } else
      return null;
  }
 
  /**
   * Set cutpoints for a single attribute.
   *
   * @param index the index of the attribute to set cutpoints for
   */
  protected void calculateCutPointsByEqualWidthBinning(int index) {

    // Scan for max and min values
    double max = 0, min = 1, currentVal;
    Instance currentInstance;
    for(int i = 0; i < getInputFormat().numInstances(); i++) {
      currentInstance = getInputFormat().instance(i);
      if (!currentInstance.isMissing(index)) {
	currentVal = currentInstance.value(index);
	if (max < min) {
	  max = min = currentVal;
	}
	if (currentVal > max) {
	  max = currentVal;
	}
	if (currentVal < min) {
	  min = currentVal;
	}
      }
    }
    double binWidth = (max - min) / m_NumBins;
    double [] cutPoints = null;
    if ((m_NumBins > 1) && (binWidth > 0)) {
      cutPoints = new double [m_NumBins - 1];
      for(int i = 1; i < m_NumBins; i++) {
	cutPoints[i - 1] = min + binWidth * i;
      }
    }
    m_CutPoints[index] = cutPoints;
  }
 
  /**
   * Set cutpoints for a single attribute.
   *
   * @param index the index of the attribute to set cutpoints for
   */
  protected void calculateCutPointsByEqualFrequencyBinning(int index) {

    // Copy data so that it can be sorted
    Instances data = new Instances(getInputFormat());

    // Sort input data
    data.sort(index);

    // Compute weight of instances without missing values
    double sumOfWeights = 0;
    for (int i = 0; i < data.numInstances(); i++) {
      if (data.instance(i).isMissing(index)) {
	break;
      } else {
	sumOfWeights += data.instance(i).weight();
      }
    }
    double freq = sumOfWeights / m_NumBins;

    // Compute break points
    double[] cutPoints = new double[m_NumBins - 1];
    double counter = 0;
    int cpindex = 0;
    for (int i = 0; i < data.numInstances() - 1; i++) {

      // Stop if value missing
      if (data.instance(i).isMissing(index)) {
	break;
      }
      counter += data.instance(i).weight();

      // Do we have a potential breakpoint?
      if (data.instance(i).value(index) < 
	  data.instance(i + 1).value(index)) {
	if (counter >= freq) {
	  cutPoints[cpindex] = (data.instance(i).value(index) +
				data.instance(i + 1).value(index)) / 2;
	  cpindex++;
	  counter = counter - freq;
	}
      }
    }

    // Did we find any cutpoints?
    if (cpindex == 0) {
      m_CutPoints[index] = null;
    } else {
      double[] cp = new double[cpindex];
      for (int i = 0; i < cpindex; i++) {
	cp[i] = cutPoints[i];
      }
      m_CutPoints[index] = cp;
    }
  }

  /**
   * Optimizes the number of bins using leave-one-out cross-validation.
   *
   * @param index the attribute index
   */
  protected void findNumBins(int index) {

    double min = Double.MAX_VALUE, max = -Double.MIN_VALUE, binWidth = 0, 
      entropy, bestEntropy = Double.MAX_VALUE, currentVal;
    double[] distribution;
    int bestNumBins  = 1;
    Instance currentInstance;

    // Find minimum and maximum
    for (int i = 0; i < getInputFormat().numInstances(); i++) {
      currentInstance = getInputFormat().instance(i);
      if (!currentInstance.isMissing(index)) {
	currentVal = currentInstance.value(index);
	if (currentVal > max) {
	  max = currentVal;
	}
	if (currentVal < min) {
	  min = currentVal;
	}
      }
    }

    // Find best number of bins
    for (int i = 0; i < m_NumBins; i++) {
      distribution = new double[i + 1];
      binWidth = (max - min) / (i + 1);

      // Compute distribution
      for (int j = 0; j < getInputFormat().numInstances(); j++) {
	currentInstance = getInputFormat().instance(j);
	if (!currentInstance.isMissing(index)) {
	  for (int k = 0; k < i + 1; k++) {
	    if (currentInstance.value(index) <= 
		(min + (((double)k + 1) * binWidth))) {
	      distribution[k] += currentInstance.weight();
	      break;
	    }
	  }
	}
      }

      // Compute cross-validated entropy
      entropy = 0;
      for (int k = 0; k < i + 1; k++) {
	if (distribution[k] < 2) {
	  entropy = Double.MAX_VALUE;
	  break;
	}
	entropy -= distribution[k] * Math.log((distribution[k] - 1) / 
					      binWidth);
      }

      // Best entropy so far?
      if (entropy < bestEntropy) {
	bestEntropy = entropy;
	bestNumBins = i + 1;
      }
    }

    // Compute cut points
    double [] cutPoints = null;
    if ((bestNumBins > 1) && (binWidth > 0)) {
      cutPoints = new double [bestNumBins - 1];
      for(int i = 1; i < bestNumBins; i++) {
	cutPoints[i - 1] = min + binWidth * i;
      }
    }
    m_CutPoints[index] = cutPoints;
   }

  /**
   * Set the output format. Takes the currently defined cutpoints and 
   * m_InputFormat and calls setOutputFormat(Instances) appropriately.
   */
  protected void setOutputFormat() {

    if (m_CutPoints == null) {
      setOutputFormat(null);
      return;
    }
    FastVector attributes = new FastVector(getInputFormat().numAttributes());
    int classIndex = getInputFormat().classIndex();
    for(int i = 0; i < getInputFormat().numAttributes(); i++) {
      if ((m_DiscretizeCols.isInRange(i)) 
	  && (getInputFormat().attribute(i).isNumeric())) {
	if (!m_MakeBinary) {
	  FastVector attribValues = new FastVector(1);
	  if (m_CutPoints[i] == null) {
	    attribValues.addElement("'All'");
	  } else {
	    for(int j = 0; j <= m_CutPoints[i].length; j++) {
	      if (j == 0) {
		attribValues.addElement("'(-inf-"
			+ Utils.doubleToString(m_CutPoints[i][j], 6) + "]'");
	      } else if (j == m_CutPoints[i].length) {
		attribValues.addElement("'("
			+ Utils.doubleToString(m_CutPoints[i][j - 1], 6) 
					+ "-inf)'");
	      } else {
		attribValues.addElement("'("
			+ Utils.doubleToString(m_CutPoints[i][j - 1], 6) + "-"
			+ Utils.doubleToString(m_CutPoints[i][j], 6) + "]'");
	      }
	    }
	  }
	  attributes.addElement(new Attribute(getInputFormat().
					      attribute(i).name(),
					      attribValues));
	} else {
	  if (m_CutPoints[i] == null) {
	    FastVector attribValues = new FastVector(1);
	    attribValues.addElement("'All'");
	    attributes.addElement(new Attribute(getInputFormat().
						attribute(i).name(),
						attribValues));
	  } else {
	    if (i < getInputFormat().classIndex()) {
	      classIndex += m_CutPoints[i].length - 1;
	    }
	    for(int j = 0; j < m_CutPoints[i].length; j++) {
	      FastVector attribValues = new FastVector(2);
	      attribValues.addElement("'(-inf-"
		      + Utils.doubleToString(m_CutPoints[i][j], 6) + "]'");
	      attribValues.addElement("'("
		      + Utils.doubleToString(m_CutPoints[i][j], 6) + "-inf)'");
	      attributes.addElement(new Attribute(getInputFormat().
						  attribute(i).name(),
						  attribValues));
	    }
	  }
	}
      } else {
	attributes.addElement(getInputFormat().attribute(i).copy());
      }
    }
    Instances outputFormat = 
      new Instances(getInputFormat().relationName(), attributes, 0);
    outputFormat.setClassIndex(classIndex);
    setOutputFormat(outputFormat);
  }

  /**
   * Convert a single instance over. The converted instance is added to 
   * the end of the output queue.
   *
   * @param instance the instance to convert
   */
  protected void convertInstance(Instance instance) {

    int index = 0;
    double [] vals = new double [outputFormatPeek().numAttributes()];
    // Copy and convert the values
    for(int i = 0; i < getInputFormat().numAttributes(); i++) {
      if (m_DiscretizeCols.isInRange(i) && 
	  getInputFormat().attribute(i).isNumeric()) {
	int j;
	double currentVal = instance.value(i);
	if (m_CutPoints[i] == null) {
	  if (instance.isMissing(i)) {
	    vals[index] = Instance.missingValue();
	  } else {
	    vals[index] = 0;
	  }
	  index++;
	} else {
	  if (!m_MakeBinary) {
	    if (instance.isMissing(i)) {
	      vals[index] = Instance.missingValue();
	    } else {
	      for (j = 0; j < m_CutPoints[i].length; j++) {
		if (currentVal <= m_CutPoints[i][j]) {
		  break;
		}
	      }
              vals[index] = j;
	    }
	    index++;
	  } else {
	    for (j = 0; j < m_CutPoints[i].length; j++) {
	      if (instance.isMissing(i)) {
                vals[index] = Instance.missingValue();
	      } else if (currentVal <= m_CutPoints[i][j]) {
                vals[index] = 0;
	      } else {
                vals[index] = 1;
	      }
	      index++;
	    }
	  }   
	}
      } else {
        vals[index] = instance.value(i);
	index++;
      }
    }
    
    Instance inst = null;
    if (instance instanceof SparseInstance) {
      inst = new SparseInstance(instance.weight(), vals);
    } else {
      inst = new Instance(instance.weight(), vals);
    }
    copyStringValues(inst, false, instance.dataset(), getInputStringIndex(),
                     getOutputFormat(), getOutputStringIndex());
    inst.setDataset(getOutputFormat());
    push(inst);
  }

  /**
   * Main method for testing this class.
   *
   * @param argv should contain arguments to the filter: use -h for help
   */
  public static void main(String [] argv) {

    try {
      if (Utils.getFlag('b', argv)) {
 	Filter.batchFilterFile(new DiscretizeFilter(), argv);
      } else {
	Filter.filterFile(new DiscretizeFilter(), argv);
      }
    } catch (Exception ex) {
      System.out.println(ex.getMessage());
    }
  }
}








