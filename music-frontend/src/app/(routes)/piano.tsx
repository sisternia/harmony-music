import { Platform, StyleSheet, View } from "react-native";
import { getApiBase } from "../index";
import StudioDOM from "@/components/dom/StudioDOM";

export default function PianoScreen() {
  if (Platform.OS === "web") return null;
  return (
    <View style={styles.screen}>
      <StudioDOM apiBase={getApiBase()} path="/piano" dom={{ style: { flex: 1 }, scrollEnabled: false }} />
    </View>
  );
}

const styles = StyleSheet.create({ screen: { flex: 1 } });
