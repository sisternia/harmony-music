import { useLocalSearchParams } from "expo-router";
import { useEffect } from "react";
import { Platform, StyleSheet, View } from "react-native";
import AuthDOM from "@/components/dom/AuthDOM";

export default function AuthScreen() {
  const { mode } = useLocalSearchParams<{ mode?: string }>();
  useEffect(() => {
    if (Platform.OS === "web") document.title = `${mode === "register" ? "Register" : "Login"} - Harmony Auralis`;
  }, [mode]);
  return (
    <View style={styles.screen}>
      <AuthDOM mode={mode === "register" ? "register" : "login"} dom={{ style: { flex: 1 } }} />
    </View>
  );
}

const styles = StyleSheet.create({ screen: { flex: 1 } });
