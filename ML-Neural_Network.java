//////////////////// ALL ASSIGNMENTS INCLUDE THIS SECTION /////////////////////
//
// Title:           Neural Networking
//
// Author:          Mudit Joshi
//
/////////////////////////////// 80 COLUMNS WIDE ///////////////////////////////


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;



/**
 * The NeuralNet class runs a multitude of methods all focused on creating a basic
 * version of a neural network that tries to accurately classify vectors based on 
 * their x values.
 * 
 * @author Mudit Joshi
 *
 */
public class NeuralNet {
	
	private static double[] NeuralNetwork;
	
	private static final String Train = "train.csv";
	private static final String Test = "test.csv";
	private static final String Eval = "eval.csv";
	
	/**
	 * Reads a csv file and returns an ArrayList of records
	 * 
	 * @param data_File
	 * @return ArrayList of Records
	 */
	private static ArrayList<Data_Storage> data_Reader(String data_File){
        ArrayList<Data_Storage> raw = new ArrayList<>();
        try (BufferedReader newline = new BufferedReader(new FileReader(data_File))) {
            String value;
            while ((value = newline.readLine()) != null) {
            	String[] split_data = value.split(",");
                double x1 = Double.parseDouble(split_data[0]);
                double x2 = Double.parseDouble(split_data[1]);
                double x3 = Double.parseDouble(split_data[2]);
                double x4 = Double.parseDouble(split_data[3]);
                double y = Double.parseDouble(split_data[4]);
                raw.add(new Data_Storage(x1, x2, x3, x4, y)); 
            }
        }
        catch(FileNotFoundException ex){
            System.out.println("No Such File");
        } catch (IOException e) {
			e.printStackTrace();
		}
        return raw;
    }
	

    
    /**
     * Simple method to call for a prediction. If the activiation
     * value is greater than 5, make the label 1. Otherwise, 0.
     * @param x
     * @return
     */
    private static int PredictFunction(double[] x) {
    	return EvalFunction(x)[5] >= 0.5 ? 1 : 0;
    }
    
    
    /**
     * Counts total correct predictions and divides that by total record
     * count
     * 
     * @param data
     * @returns accuracy value
     */
    private static double Precision(ArrayList<Data_Storage> data) {
        int num_Predict = 0;
        for (Data_Storage new_data : data) {
            int new_predict = PredictFunction(new_data.x);
            int temp = (int) new_data.y;
            if (new_predict == temp) {
                num_Predict++;
            }
        }
        return (double) num_Predict / data.size();
    }

    
    
    
    /**
     * Sums up the total error value for a records list
     * @param rawvalue
     * @returns setError value
     */
    private static double ErrorFunction(ArrayList<Data_Storage> rawvalue) {
    	double val_Error = 0.0;
    	for (Data_Storage value_Rec : rawvalue) {
    		val_Error += Math.pow(EvalFunction(value_Rec.x)[5] - value_Rec.y, 2);
    	}
    	
    	val_Error *= 0.5;
    	return val_Error ;
    }
    
    
    
	/**
	 * A simple print method for any double arrays that need to 
	 * be printed.
	 * 
	 * @param rawdata
	 */
    private static void Result(double[] rawdata) {
        StringBuilder values = new StringBuilder();
        for (double data : rawdata) {
            values.append(String.format("%.5f ", data));
        }
        System.out.println(values.toString().trim());
    }
	
	
	
    private static double[] HiddenFunction(double[] new_values, double y_value) {
    	double output_Val = OutputFunction(new_values[5], y_value); 
    	
    	double newEDV_Value = NeuralNetwork[11] * output_Val;
    	double value = newEDV_Value * new_values[1];
    	value *= (1 - new_values[1]);
    	
    	double EDV_Value2 = NeuralNetwork[12] * output_Val;
    	double value2 = EDV_Value2 * new_values[3];
    	value2 *= (1 - new_values[3]);
    	
    	return new double [] {value, value2};
    	
    	
    	
    }


    
    /**
     * Updates the weights based off their partial derivatives of error
     * 
     * @param weighted_value
     * @param Eta_value
     * @return
     */
    private static void SDFuntion(double[] weighted_value, double Eta_value) {
    	for (int i = 0; i < NeuralNetwork.length; i++) {
    		NeuralNetwork[i] -= Eta_value * weighted_value[i];
    	}
    }
    
    


    /**
     * Evaluates the Sigmoid activation of the internal and output units
     * 
     * @return uA, vA, uB, vB, uC, vC
     */
    private static double[] EvalFunction(double[] xvalue_array) {
        double uA_Value = NeuralNetwork[0] + NeuralNetwork[1] * xvalue_array[0] + NeuralNetwork[2] * xvalue_array[1] + NeuralNetwork[3] * xvalue_array[2] + NeuralNetwork[4] * xvalue_array[3];
        double vA_Value = SigmoidFunction(uA_Value);
        double uB_Value = NeuralNetwork[5] + NeuralNetwork[6] * xvalue_array[0] + NeuralNetwork[7] * xvalue_array[1] + NeuralNetwork[8] * xvalue_array[2] + NeuralNetwork[9] * xvalue_array[3];
        double vB_Value = SigmoidFunction (uB_Value);
        double uC_Value = NeuralNetwork[10] + NeuralNetwork[11] * vA_Value + NeuralNetwork[12] * vB_Value;
        double vC_Value = SigmoidFunction(uC_Value);
        return new double[] {uA_Value, vA_Value, uB_Value, vB_Value, uC_Value, vC_Value};
    }
    
    
	
    /**
     * 
     * @param x_value = x values from a given vector
     * @param new_values = evaluated values 
     * @param y_value = label
     * @returns the partial derivative of all units in respect to their weights
     */
    private static double[] WeightedFunction(double[] x_value, double[] new_values, double y_value) {
    	double value_Output = OutputFunction(new_values[5], y_value);
    	double[] value_Hidden = HiddenFunction(new_values, y_value);
    	double[] newEDW_Value = new double[NeuralNetwork.length];
    	
    	// Output layer 
    	newEDW_Value[12] = value_Output * new_values[3];
    	newEDW_Value[11] = value_Output * new_values[1];
    	newEDW_Value[10] = value_Output; // w10(3)
    	
    	// a2 inner layer 
    	newEDW_Value[9] = value_Hidden[1] * x_value[3];
    	newEDW_Value[8] = value_Hidden[1] * x_value[2];
    	newEDW_Value[7] = value_Hidden[1] * x_value[1];
    	newEDW_Value[6] = value_Hidden[1] * x_value[0];
    	newEDW_Value[5] = value_Hidden[1]; // w20
    	
    	// a1 inner layer
    	newEDW_Value[4] = value_Hidden[0] * x_value[3];
    	newEDW_Value[3] = value_Hidden[0] * x_value[2];
    	newEDW_Value[2] = value_Hidden[0] * x_value[1];
    	newEDW_Value[1] = value_Hidden[0] * x_value[0];
    	newEDW_Value[0] = value_Hidden[0]; // w10
    	
    	return newEDW_Value;
    }
    
   
    
    /**
     * 
     * @param vC_Value = activation value
     * @param y_Value = label
     * @returns the partial derivative of error  
     */
    private static double OutputFunction(double vC_Value, double y_Value) {
    	return (vC_Value - y_Value) * vC_Value * (1 - vC_Value); 
    }
    
    
    
	/**
	 * Sigmoid function on any value 
	 * 
	 * @param x_value
	 * @returns Sigmoid activation value
	 */
	private static double SigmoidFunction(double x_value) {
		return 1.0 / (1 + Math.exp(-x_value));
	}

	
	/**
	 * The main method uses the given arguments of the flag, weights, x values,
	 * and or an eta value to choose the methods and solutions desired.
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		
		double y_value = Double.NaN;
		double Val_Eta = Double.NaN;
		int Flag = Integer.parseInt(args[0]);
		
		// Initialize and store all weights
		NeuralNetwork = new double[13];
		for (int i = 0; i < NeuralNetwork.length; i++) {
			NeuralNetwork[i] = Double.parseDouble(args[i + 1]);
		}
		
		// Initialize and store any potential x values
		if (Flag >= 100 && Flag <= 400) {
			double[] x_storage = new double[4];
            for (int i = 0; i < x_storage.length; i++) {
                x_storage[i] = Double.parseDouble(args[i + NeuralNetwork.length + 1]);
            }
            
            // Create a y                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
            if (Flag >= 200) {
            	y_value = Double.parseDouble(args[NeuralNetwork.length + x_storage.length + 1]);
            }
			
            // For the 100 Flag
            if (Flag == 100) {
				double[] eval_value = EvalFunction(x_storage);
				Result(new double[] {eval_value[1], eval_value[3]});
				System.out.println(String.format("%.5f", eval_value[5]));
            }
            
            // For the 200 Flag
            if (Flag == 200) {
                double[] eval_value = EvalFunction(x_storage);
    			double error_value = OutputFunction(eval_value[5], y_value);
    			System.out.println(String.format("%.5f", error_value));
    		}
			
            // For the 300 Flag
            if (Flag == 300) {
    			double[] hiddenvalue = HiddenFunction(EvalFunction(x_storage), y_value);
    			Result(hiddenvalue);
    			
    		}
            
            // For the 400 Flag
            if (Flag == 400) {
            	double[] weightedvalue = WeightedFunction(x_storage, EvalFunction(x_storage), y_value);
            	Result(new double[] {weightedvalue[10], weightedvalue[11], weightedvalue[12]});
            	Result(new double[] {weightedvalue[0], weightedvalue[1], weightedvalue[2], weightedvalue[3], weightedvalue[4]});
            	Result(new double[] {weightedvalue[5], weightedvalue[6], weightedvalue[7], weightedvalue[8], weightedvalue[9]});
            	
            	
            }
		} else {
			
			// For 500 and above Flag
			Val_Eta = Double.parseDouble(args[NeuralNetwork.length + 1]);
        	ArrayList<Data_Storage> train = data_Reader(Train);
        	ArrayList<Data_Storage> eval = data_Reader(Eval);
        	ArrayList<Data_Storage> test_function = data_Reader(Test);
 
        	// For the flag 500
        	if (Flag == 500) {
	        	for (Data_Storage rec_data : train) {
	        		double[] temp_storage = WeightedFunction(rec_data.x, EvalFunction(rec_data.x), rec_data.y);
	        		SDFuntion(temp_storage, Val_Eta);
	        		Result(NeuralNetwork);
	        		Result(new double[] {ErrorFunction(eval)});
	            }
        	}
			
        	// For the Flag 600
        	if (Flag == 600) {
        		int iterator;
                double prev_Error = Double.POSITIVE_INFINITY;
                double Error_value = Double.NEGATIVE_INFINITY;
                for (Data_Storage record : train) {
                    SDFuntion(WeightedFunction(record.x, EvalFunction(record.x), record.y), Val_Eta);
                }
                Error_value = ErrorFunction(eval); 
                
                for (Data_Storage rec_data : test_function) {
                	System.out.print((int) rec_data.y + " " + PredictFunction(rec_data.x) + " " + 
                			String.format("%.5f", EvalFunction(rec_data.x)[5]));
                	System.out.println();
                }
                System.out.println(String.format("%.2f", Precision(test_function)));
        	}
			
		}
	}
}


/**
 * Class to store individual records within the csv files
 * @author Mudit Joshi
 *
 */
class Data_Storage {
	public double[] x;
	public double y;
	
	public Data_Storage(double x1, double x2, double x3, double x4, double y) {
		x = new double[4];
		x[0] = x1;
		x[1] = x2;
		x[2] = x3;
		x[3] = x4;
		this.y = y;
	}
}
