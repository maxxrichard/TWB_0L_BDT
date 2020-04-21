#include "VLQWbLepOffline/XGBoost.h"
#include "TTHbbToolManager/ToolManager.h"
#include "TTHbbObjects/TTHbbUtils.h"
#include "TTHbbConfiguration/GlobalConfiguration.h"
#include <xgboost/c_api.h>
#include <dmlc/io.h>

#include <chrono>

#include <string>
//Include the headers needed for sequential model
#include <fstream>
#include <iostream>
#include <experimental/filesystem>
#include <algorithm>    // std::min_element, std::max_element
namespace fs = std::experimental::filesystem;
namespace ch = std::chrono;



namespace TTHbb{
    
//____________________________________________________

    XGBoostClass::XGBoostClass() {


    }
//____________________________________________________
    
    XGBoostClass::~XGBoostClass() {
    
    }
//____________________________________________________
//____________________________________________________

//____________________________________________________

    void XGBoostClass::initialise() {

        auto* config = TTHbb::GlobalConfiguration::get();
        std::string input((*config)("XGBoostClass.ModelFolder"));
        std::cout << "using Model Folder: " << input << std::endl;

        wps = TTHbb::util::vectoriseString((*config)("MVAVariables.bTagWPs"));

        for (auto & p : fs::directory_iterator(input)){
            std::string mod_name = p.path().string();
            if (mod_name.find(".model") != std::string::npos){

                for (auto w : wps){
                    if (mod_name.find(w) != std::string::npos){
                        
                        BoosterHandle booster;
                        XGBoosterCreate(0,0,&booster);
                        int x = XGBoosterLoadModel(booster, mod_name.c_str());
                        if (x == 0) {
                            std::cout << "Loaded model: " << mod_name << std::endl;
                        } 
                        else std::cout << "Failed to load model: " << mod_name << std::endl;
                       
                        std::string mod_label = fs::path(mod_name).filename(); 
                        models[mod_label.substr(0, 10)] = booster; 
                    }
                }   
            }
        }

       for (auto m : models){
            std::string w = m.first;

            std::ifstream varfile(input + w.substr(0, 3) + std::string(".train_vars")); 
            std::vector <std::string> vars = {};

            std::string var;

            std::cout << "Using vars: " << std::endl;
            while (varfile >> var){
                std::cout << var << std::endl;
                vars.push_back(var);
            }
            var_map[w] = vars;
        }

       std::cout << "INFO in XGBoost: XGBoost Initialized" << std::endl;
    }
//____________________________________________________

    void XGBoostClass::printBDTInfo() {


    }
//____________________________________________________

    void XGBoostClass::apply(TTHbb::Event* event) {
        
        
        for (auto m : models){ 

            std::string w = m.first;
            std::vector<std::string> vars = var_map[w];
            BoosterHandle h_booster = m.second; 
              
            n_feat = vars.size();
     
            for (const auto& var: vars) {
                if(event->checkIntVariable(var)){
                    inputs[var] = event->intVariable(var);
                } else
                if(event->checkFloatVariable(var)){
                    inputs[var] = event->floatVariable(var);
                } else
                if (event->checkCharVariable(var)){
                    inputs[var] = event->charVariable(var);
                } else
                    std::cout << "Can't find var " << var << std::endl;
            }
           

            int rows = 1;
            int cols = n_feat;


            float dummy[1][n_feat] = {{0}};

            int i = 0;
            for (const auto& in: vars){
                dummy[0][i] =  inputs[in];
                i++;
            } 

    
            int x = XGDMatrixCreateFromMat((float *)dummy, rows, cols , -999.9f, &h_test);




            bst_ulong out_len;
            const float *f;   
            XGBoosterPredict(h_booster, h_test, 0,0,&out_len,&f);

            std::string outputName = "XGBoost_" + w;
            if (out_len == 1){
                event->floatVariable(outputName) = f[0];

            } else {
                    
                event->intVariable(outputName + "_Max_Class") = std::distance(f, std::max_element(f,f + out_len));               
                for (int class_id = 0; class_id <  out_len; class_id++){ 
                
                    event->floatVariable(outputName + "_" + std::to_string(class_id)) = f[class_id];
                }
            }

        }
       
    }
//____________________________________________________

    void XGBoostClass::defaultValues(TTHbb::Event* event) const {
   }
//____________________________________________________
    
    void XGBoostClass::finalise() {
    }
//____________________________________________________
}
